import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
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


class GradientBoostingRegressorPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Gradient Boosting Regressor Plugin - Sequential Boosting
    
    Gradient Boosting is a powerful ensemble method that builds models sequentially,
    where each subsequent model learns from the mistakes of the previous ones by
    fitting to the residuals. It combines weak learners (typically decision trees)
    to create a strong predictive model through iterative improvement.
    
    Key Features:
    - Sequential learning: Each tree improves upon previous predictions
    - Residual fitting: Models learn from prediction errors
    - Gradient-based optimization: Uses gradient descent principles
    - Flexible loss functions: Supports various regression objectives
    - Built-in regularization: Early stopping and shrinkage
    - Feature importance: Comprehensive importance analysis
    - Learning trajectory: Monitor training progress over iterations
    - Overfitting control: Multiple regularization mechanisms
    - Advanced boosting analysis: Detailed insights into boosting process
    - Comprehensive performance monitoring: Training vs validation curves
    - Extensive hyperparameter sensitivity analysis
    - Computational efficiency profiling
    """
    
    def __init__(
        self,
        # Core boosting parameters
        n_estimators=100,
        learning_rate=0.1,
        loss='squared_error',
        
        # Tree parameters
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        
        # Advanced parameters
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        subsample=1.0,
        
        # Regularization
        alpha=0.9,  # For Huber and Quantile loss
        
        # Control parameters
        random_state=42,
        verbose=0,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        
        # Analysis options
        compute_feature_importance=True,
        compute_permutation_importance=True,
        boosting_analysis=True,
        learning_trajectory_analysis=True,
        residual_analysis=True,
        
        # Advanced analysis
        staged_prediction_analysis=True,
        feature_interaction_analysis=True,
        learning_curve_analysis=True,
        hyperparameter_sensitivity_analysis=True,
        computational_efficiency_analysis=True,
        
        # Boosting specific analysis
        early_stopping_analysis=True,
        overfitting_analysis=True,
        shrinkage_analysis=True,
        loss_function_analysis=True,
        
        # Comparison analysis
        compare_with_random_forest=True,
        compare_with_linear_model=True,
        ensemble_evolution_analysis=True,
        
        # Visualization options
        visualize_trees=False,
        max_trees_to_visualize=3,
        feature_importance_analysis=True,
        prediction_distribution_analysis=True,
        
        # Performance analysis
        cross_validation_analysis=True,
        cv_folds=5,
        performance_benchmarking=True,
        
        # Gradient boosting specific analysis
        gradient_analysis=True,
        bias_variance_decomposition=True,
        convergence_analysis=True
    ):
        super().__init__()
        
        # Core boosting parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        
        # Tree parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        
        # Advanced parameters
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.subsample = subsample
        
        # Regularization
        self.alpha = alpha
        
        # Control parameters
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        
        # Analysis options
        self.compute_feature_importance = compute_feature_importance
        self.compute_permutation_importance = compute_permutation_importance
        self.boosting_analysis = boosting_analysis
        self.learning_trajectory_analysis = learning_trajectory_analysis
        self.residual_analysis = residual_analysis
        
        # Advanced analysis
        self.staged_prediction_analysis = staged_prediction_analysis
        self.feature_interaction_analysis = feature_interaction_analysis
        self.learning_curve_analysis = learning_curve_analysis
        self.hyperparameter_sensitivity_analysis = hyperparameter_sensitivity_analysis
        self.computational_efficiency_analysis = computational_efficiency_analysis
        
        # Boosting specific analysis
        self.early_stopping_analysis = early_stopping_analysis
        self.overfitting_analysis = overfitting_analysis
        self.shrinkage_analysis = shrinkage_analysis
        self.loss_function_analysis = loss_function_analysis
        
        # Comparison analysis
        self.compare_with_random_forest = compare_with_random_forest
        self.compare_with_linear_model = compare_with_linear_model
        self.ensemble_evolution_analysis = ensemble_evolution_analysis
        
        # Visualization options
        self.visualize_trees = visualize_trees
        self.max_trees_to_visualize = max_trees_to_visualize
        self.feature_importance_analysis = feature_importance_analysis
        self.prediction_distribution_analysis = prediction_distribution_analysis
        
        # Performance analysis
        self.cross_validation_analysis = cross_validation_analysis
        self.cv_folds = cv_folds
        self.performance_benchmarking = performance_benchmarking
        
        # Gradient boosting specific analysis
        self.gradient_analysis = gradient_analysis
        self.bias_variance_decomposition = bias_variance_decomposition
        self.convergence_analysis = convergence_analysis
        
        # Required plugin metadata
        self._name = "Gradient Boosting Regressor"
        self._description = "Sequential boosting ensemble with iterative residual learning"
        self._category = "Ensemble Methods"
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 50
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        
        # Analysis results
        self.feature_importance_analysis_ = {}
        self.boosting_analysis_ = {}
        self.learning_trajectory_analysis_ = {}
        self.residual_analysis_ = {}
        self.staged_prediction_analysis_ = {}
        self.feature_interaction_analysis_ = {}
        self.learning_curve_analysis_ = {}
        self.hyperparameter_sensitivity_analysis_ = {}
        self.computational_efficiency_analysis_ = {}
        self.early_stopping_analysis_ = {}
        self.overfitting_analysis_ = {}
        self.shrinkage_analysis_ = {}
        self.loss_function_analysis_ = {}
        self.random_forest_comparison_ = {}
        self.linear_model_comparison_ = {}
        self.ensemble_evolution_analysis_ = {}
        self.cross_validation_analysis_ = {}
        self.performance_benchmarking_ = {}
        self.gradient_analysis_ = {}
        self.bias_variance_decomposition_ = {}
        self.convergence_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Gradient Boosting Regressor with comprehensive analysis
        
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
        
        # Create and configure Gradient Boosting model
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            subsample=self.subsample,
            alpha=self.alpha,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            ccp_alpha=self.ccp_alpha
        )
        
        # Fit the model
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        # Perform comprehensive analysis
        self._analyze_feature_importance()
        self._analyze_boosting_process()
        self._analyze_learning_trajectory()
        self._analyze_residuals()
        self._analyze_staged_predictions()
        self._analyze_feature_interactions()
        self._analyze_learning_curves()
        self._analyze_hyperparameter_sensitivity()
        self._analyze_computational_efficiency()
        self._analyze_early_stopping()
        self._analyze_overfitting()
        self._analyze_shrinkage_effects()
        self._analyze_loss_function()
        self._compare_with_random_forest()
        self._compare_with_linear_model()
        self._analyze_ensemble_evolution()
        self._analyze_cross_validation()
        self._analyze_performance_benchmarks()
        self._analyze_gradients()
        self._analyze_bias_variance_decomposition()
        self._analyze_convergence()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted Gradient Boosting model
        
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
        return self.model_.predict(X)
    
    def predict_staged(self, X, n_estimators=None):
        """
        Make staged predictions showing evolution of predictions
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        n_estimators : int, optional
            Number of estimators to use (default: all)
        
        Returns:
        --------
        staged_predictions : generator
            Generator yielding predictions at each stage
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        if n_estimators is None:
            return self.model_.staged_predict(X)
        else:
            # Get predictions up to n_estimators
            staged_preds = []
            for i, pred in enumerate(self.model_.staged_predict(X)):
                if i >= n_estimators:
                    break
                staged_preds.append(pred)
            return staged_preds
    
    def predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty estimates using staged predictions
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        
        Returns:
        --------
        results : dict
            Dictionary containing predictions, uncertainty estimates, and staged predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Get staged predictions
        staged_predictions = list(self.model_.staged_predict(X))
        staged_array = np.array(staged_predictions)
        
        # Final predictions
        final_predictions = staged_array[-1]
        
        # Calculate prediction evolution statistics
        prediction_std = np.std(staged_array, axis=0)
        prediction_range = np.max(staged_array, axis=0) - np.min(staged_array, axis=0)
        
        # Early vs late predictions
        early_predictions = np.mean(staged_array[:len(staged_array)//4], axis=0)
        late_predictions = np.mean(staged_array[-len(staged_array)//4:], axis=0)
        convergence_measure = np.abs(late_predictions - early_predictions)
        
        # Uncertainty based on prediction stability
        uncertainty_score = prediction_std / (np.abs(final_predictions) + 1e-10)
        
        # Confidence intervals based on boosting evolution
        confidence_95_lower = final_predictions - 1.96 * prediction_std
        confidence_95_upper = final_predictions + 1.96 * prediction_std
        
        return {
            'predictions': final_predictions,
            'staged_predictions': staged_array,
            'prediction_std': prediction_std,
            'prediction_range': prediction_range,
            'uncertainty_score': uncertainty_score,
            'convergence_measure': convergence_measure,
            'confidence_95_lower': confidence_95_lower,
            'confidence_95_upper': confidence_95_upper,
            'prediction_interval_width': confidence_95_upper - confidence_95_lower,
            'boosting_stability': 1.0 / (1.0 + uncertainty_score),
            'early_predictions': early_predictions,
            'late_predictions': late_predictions,
            'n_estimators_used': len(staged_predictions)
        }
    
    def _analyze_feature_importance(self):
        """Analyze feature importance with Gradient Boosting specific insights"""
        if not self.compute_feature_importance:
            return
        
        try:
            # Built-in feature importance (mean decrease in impurity)
            builtin_importance = self.model_.feature_importances_
            
            # Permutation importance if requested
            permutation_imp = None
            permutation_imp_std = None
            if self.compute_permutation_importance:
                try:
                    perm_imp_result = permutation_importance(
                        self.model_, self.X_original_, self.y_original_,
                        n_repeats=10, random_state=self.random_state, 
                        scoring='neg_mean_squared_error', n_jobs=-1
                    )
                    permutation_imp = perm_imp_result.importances_mean
                    permutation_imp_std = perm_imp_result.importances_std
                except:
                    permutation_imp = None
                    permutation_imp_std = None
            
            # Gradient Boosting specific: Staged feature importance
            staged_importance = self._calculate_staged_feature_importance()
            
            # Feature importance evolution analysis
            importance_evolution = self._analyze_importance_evolution(staged_importance)
            
            # Feature selection based on boosting
            boosting_feature_selection = self._analyze_boosting_feature_selection(
                builtin_importance, staged_importance
            )
            
            # Feature importance ranking
            importance_ranking = np.argsort(builtin_importance)[::-1]
            
            # Calculate importance statistics
            importance_stats = {
                'mean_importance': np.mean(builtin_importance),
                'std_importance': np.std(builtin_importance),
                'max_importance': np.max(builtin_importance),
                'min_importance': np.min(builtin_importance),
                'importance_concentration': self._calculate_importance_concentration(builtin_importance),
                'effective_features': np.sum(builtin_importance > np.mean(builtin_importance)),
                'gini_coefficient': self._calculate_gini_coefficient(builtin_importance),
                'boosting_stability': importance_evolution.get('stability_score', 0.0)
            }
            
            self.feature_importance_analysis_ = {
                'builtin_importance': builtin_importance,
                'permutation_importance': permutation_imp,
                'permutation_importance_std': permutation_imp_std,
                'staged_importance': staged_importance,
                'importance_evolution': importance_evolution,
                'boosting_feature_selection': boosting_feature_selection,
                'feature_ranking': importance_ranking,
                'feature_names': self.feature_names_,
                'importance_statistics': importance_stats,
                'top_features': [
                    {
                        'name': self.feature_names_[i],
                        'builtin_importance': builtin_importance[i],
                        'permutation_importance': permutation_imp[i] if permutation_imp is not None else None,
                        'evolution_stability': importance_evolution.get('feature_stability', {}).get(i, 0.0),
                        'boosting_selection_score': boosting_feature_selection.get('selection_scores', [0.0] * len(self.feature_names_))[i]
                    }
                    for i in importance_ranking[:15]
                ]
            }
            
        except Exception as e:
            self.feature_importance_analysis_ = {
                'error': f'Could not analyze feature importance: {str(e)}'
            }
    
    def _calculate_staged_feature_importance(self):
        """Calculate feature importance at different stages of boosting"""
        try:
            # Sample stages for analysis
            n_stages = min(20, self.model_.n_estimators_)
            stage_indices = np.linspace(0, self.model_.n_estimators_ - 1, n_stages, dtype=int)
            
            staged_importance = []
            
            for stage in stage_indices:
                # Create temporary model with limited estimators
                temp_model = GradientBoostingRegressor(
                    n_estimators=stage + 1,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
                
                try:
                    temp_model.fit(self.X_original_, self.y_original_)
                    staged_importance.append(temp_model.feature_importances_)
                except:
                    # If training fails, use zero importance
                    staged_importance.append(np.zeros(self.n_features_in_))
            
            return {
                'stage_indices': stage_indices,
                'importance_matrix': np.array(staged_importance),
                'n_stages_analyzed': len(stage_indices)
            }
            
        except:
            return {'error': 'Could not calculate staged importance'}
    
    def _analyze_importance_evolution(self, staged_importance):
        """Analyze how feature importance evolves during boosting"""
        try:
            if 'error' in staged_importance:
                return {'error': 'No staged importance data'}
            
            importance_matrix = staged_importance['importance_matrix']
            
            # Calculate stability metrics for each feature
            feature_stability = {}
            for i in range(self.n_features_in_):
                feature_importance_series = importance_matrix[:, i]
                
                # Coefficient of variation
                cv = np.std(feature_importance_series) / (np.mean(feature_importance_series) + 1e-10)
                
                # Trend analysis
                stages = np.arange(len(feature_importance_series))
                correlation = np.corrcoef(stages, feature_importance_series)[0, 1] if len(stages) > 1 else 0
                
                feature_stability[i] = {
                    'cv': cv,
                    'trend_correlation': correlation,
                    'final_vs_initial': feature_importance_series[-1] / (feature_importance_series[0] + 1e-10),
                    'stability_score': 1.0 / (1.0 + cv)
                }
            
            # Overall evolution characteristics
            overall_stability = np.mean([fs['stability_score'] for fs in feature_stability.values()])
            
            return {
                'feature_stability': feature_stability,
                'overall_stability': overall_stability,
                'stability_score': overall_stability,
                'interpretation': 'Higher stability indicates consistent feature importance across boosting iterations'
            }
            
        except:
            return {'error': 'Could not analyze importance evolution'}
    
    def _analyze_boosting_feature_selection(self, builtin_importance, staged_importance):
        """Analyze feature selection based on boosting progression"""
        try:
            if 'error' in staged_importance:
                return {'error': 'No staged importance data'}
            
            importance_matrix = staged_importance['importance_matrix']
            
            # Features that gain importance over time
            early_importance = np.mean(importance_matrix[:len(importance_matrix)//3], axis=0)
            late_importance = np.mean(importance_matrix[-len(importance_matrix)//3:], axis=0)
            
            importance_gain = late_importance - early_importance
            
            # Selection scores combining final importance and stability
            selection_scores = []
            for i in range(self.n_features_in_):
                final_importance = builtin_importance[i]
                stability = 1.0 / (1.0 + np.std(importance_matrix[:, i]))
                gain = importance_gain[i]
                
                # Composite score
                selection_score = final_importance * stability * (1.0 + max(0, gain))
                selection_scores.append(selection_score)
            
            selection_scores = np.array(selection_scores)
            
            # Recommended features
            selection_threshold = np.mean(selection_scores) + 0.5 * np.std(selection_scores)
            recommended_features = selection_scores > selection_threshold
            
            return {
                'selection_scores': selection_scores,
                'early_importance': early_importance,
                'late_importance': late_importance,
                'importance_gain': importance_gain,
                'recommended_features': recommended_features,
                'selection_threshold': selection_threshold,
                'n_recommended': np.sum(recommended_features)
            }
            
        except:
            return {'error': 'Could not analyze boosting feature selection'}
    
    def _calculate_importance_concentration(self, importance_values):
        """Calculate how concentrated the importance is in top features"""
        try:
            sorted_importance = np.sort(importance_values)[::-1]
            total_importance = np.sum(sorted_importance)
            
            # Calculate cumulative importance
            cumsum = np.cumsum(sorted_importance) / total_importance
            
            # Find features needed for different importance thresholds
            features_80 = np.argmax(cumsum >= 0.8) + 1
            features_90 = np.argmax(cumsum >= 0.9) + 1
            
            return {
                'top_5_concentration': np.sum(sorted_importance[:5]) / total_importance,
                'top_10_concentration': np.sum(sorted_importance[:10]) / total_importance,
                'features_for_80_percent': features_80,
                'features_for_90_percent': features_90,
                'concentration_ratio': features_80 / len(importance_values)
            }
        except:
            return {'error': 'Could not calculate concentration'}
    
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for importance distribution"""
        try:
            sorted_values = np.sort(values)
            n = len(values)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        except:
            return 0.0
    
    def _analyze_boosting_process(self):
        """Analyze the sequential boosting process"""
        if not self.boosting_analysis:
            return
        
        try:
            # Training scores over iterations
            train_scores = self.model_.train_score_
            
            # Validation scores if available
            validation_scores = None
            if hasattr(self.model_, 'validation_score_'):
                validation_scores = self.model_.validation_score_
            
            # Analyze training progression
            progression_analysis = self._analyze_training_progression(train_scores, validation_scores)
            
            # Sequential improvement analysis
            improvement_analysis = self._analyze_sequential_improvement(train_scores)
            
            # Boosting convergence
            convergence_analysis = self._analyze_boosting_convergence(train_scores, validation_scores)
            
            # Estimator utilization
            estimator_analysis = self._analyze_estimator_utilization()
            
            self.boosting_analysis_ = {
                'train_scores': train_scores,
                'validation_scores': validation_scores,
                'progression_analysis': progression_analysis,
                'improvement_analysis': improvement_analysis,
                'convergence_analysis': convergence_analysis,
                'estimator_analysis': estimator_analysis,
                'n_estimators_used': self.model_.n_estimators_,
                'final_train_score': train_scores[-1] if len(train_scores) > 0 else None
            }
            
        except Exception as e:
            self.boosting_analysis_ = {
                'error': f'Could not analyze boosting process: {str(e)}'
            }
    
    def _analyze_training_progression(self, train_scores, validation_scores):
        """Analyze the progression of training scores"""
        try:
            if len(train_scores) == 0:
                return {'error': 'No training scores available'}
            
            # Basic statistics
            initial_score = train_scores[0]
            final_score = train_scores[-1]
            total_improvement = final_score - initial_score
            
            # Rate of improvement
            improvements = np.diff(train_scores)
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            
            # Early vs late improvement
            early_improvements = improvements[:len(improvements)//3]
            late_improvements = improvements[-len(improvements)//3:]
            
            early_mean = np.mean(early_improvements) if len(early_improvements) > 0 else 0
            late_mean = np.mean(late_improvements) if len(late_improvements) > 0 else 0
            
            # Validation progression if available
            validation_analysis = {}
            if validation_scores is not None:
                val_improvements = np.diff(validation_scores)
                validation_analysis = {
                    'initial_val_score': validation_scores[0],
                    'final_val_score': validation_scores[-1],
                    'val_total_improvement': validation_scores[-1] - validation_scores[0],
                    'val_mean_improvement': np.mean(val_improvements),
                    'generalization_gap': train_scores[-1] - validation_scores[-1]
                }
            
            return {
                'initial_score': initial_score,
                'final_score': final_score,
                'total_improvement': total_improvement,
                'mean_improvement_per_iteration': mean_improvement,
                'improvement_stability': std_improvement,
                'early_improvement_rate': early_mean,
                'late_improvement_rate': late_mean,
                'learning_efficiency': early_mean / (late_mean + 1e-10),
                'validation_analysis': validation_analysis
            }
            
        except:
            return {'error': 'Could not analyze training progression'}
    
    def _analyze_sequential_improvement(self, train_scores):
        """Analyze sequential improvement characteristics"""
        try:
            if len(train_scores) < 5:
                return {'error': 'Insufficient training scores for analysis'}
            
            improvements = np.diff(train_scores)
            
            # Improvement pattern analysis
            positive_improvements = improvements > 0
            negative_improvements = improvements < 0
            
            # Consecutive improvements/deteriorations
            improvement_streaks = []
            current_streak = 0
            for imp in positive_improvements:
                if imp:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        improvement_streaks.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                improvement_streaks.append(current_streak)
            
            # Improvement characteristics
            improvement_characteristics = {
                'improvement_ratio': np.mean(positive_improvements),
                'deterioration_ratio': np.mean(negative_improvements),
                'mean_positive_improvement': np.mean(improvements[positive_improvements]) if np.any(positive_improvements) else 0,
                'mean_negative_improvement': np.mean(improvements[negative_improvements]) if np.any(negative_improvements) else 0,
                'max_improvement_streak': max(improvement_streaks) if improvement_streaks else 0,
                'improvement_consistency': np.std(improvements[positive_improvements]) if np.any(positive_improvements) else 0
            }
            
            # Diminishing returns analysis
            window_size = min(10, len(improvements) // 4)
            if window_size > 0:
                early_window = np.mean(improvements[:window_size])
                late_window = np.mean(improvements[-window_size:])
                diminishing_returns_ratio = late_window / (early_window + 1e-10)
            else:
                diminishing_returns_ratio = 1.0
            
            return {
                'improvement_characteristics': improvement_characteristics,
                'diminishing_returns_ratio': diminishing_returns_ratio,
                'sequential_efficiency': improvement_characteristics['improvement_ratio'],
                'learning_stability': 1.0 / (1.0 + improvement_characteristics['improvement_consistency'])
            }
            
        except:
            return {'error': 'Could not analyze sequential improvement'}
    
    def _analyze_boosting_convergence(self, train_scores, validation_scores):
        """Analyze convergence characteristics of boosting"""
        try:
            # Training convergence
            train_convergence = self._check_convergence(train_scores)
            
            # Validation convergence
            val_convergence = {}
            if validation_scores is not None:
                val_convergence = self._check_convergence(validation_scores)
            
            # Early stopping analysis
            early_stopping_analysis = {}
            if validation_scores is not None:
                early_stopping_analysis = self._analyze_early_stopping_point(
                    train_scores, validation_scores
                )
            
            # Optimal number of estimators
            optimal_n_estimators = self._find_optimal_n_estimators(train_scores, validation_scores)
            
            return {
                'train_convergence': train_convergence,
                'validation_convergence': val_convergence,
                'early_stopping_analysis': early_stopping_analysis,
                'optimal_n_estimators': optimal_n_estimators,
                'converged': train_convergence.get('converged', False),
                'overfitting_detected': early_stopping_analysis.get('overfitting_detected', False)
            }
            
        except:
            return {'error': 'Could not analyze boosting convergence'}
    
    def _check_convergence(self, scores):
        """Check if scores have converged"""
        try:
            if len(scores) < 10:
                return {'converged': False, 'reason': 'Insufficient iterations'}
            
            # Check last 10 iterations for stability
            recent_scores = scores[-10:]
            score_std = np.std(recent_scores)
            score_mean = np.mean(recent_scores)
            
            # Relative stability
            cv = score_std / (abs(score_mean) + 1e-10)
            
            # Trend analysis
            iterations = np.arange(len(recent_scores))
            if len(iterations) > 1:
                correlation = abs(np.corrcoef(iterations, recent_scores)[0, 1])
            else:
                correlation = 0
            
            # Convergence criteria
            converged = cv < 0.01 and correlation < 0.3
            
            return {
                'converged': converged,
                'cv': cv,
                'trend_correlation': correlation,
                'stability_score': 1.0 / (1.0 + cv),
                'recent_std': score_std,
                'convergence_iteration': len(scores) - 10 if converged else None
            }
            
        except:
            return {'converged': False, 'error': 'Could not check convergence'}
    
    def _analyze_early_stopping_point(self, train_scores, validation_scores):
        """Analyze optimal early stopping point"""
        try:
            if len(validation_scores) < 5:
                return {'error': 'Insufficient validation scores'}
            
            # Find best validation score
            best_val_idx = np.argmax(validation_scores)
            best_val_score = validation_scores[best_val_idx]
            
            # Overfitting analysis
            overfitting_start = None
            for i in range(best_val_idx + 1, len(validation_scores)):
                if validation_scores[i] < best_val_score - 0.01:  # Significant drop
                    overfitting_start = i
                    break
            
            # Generalization gap evolution
            gaps = [train_scores[i] - validation_scores[i] for i in range(len(validation_scores))]
            gap_trend = np.polyfit(range(len(gaps)), gaps, 1)[0] if len(gaps) > 1 else 0
            
            return {
                'optimal_n_estimators': best_val_idx + 1,
                'best_validation_score': best_val_score,
                'overfitting_detected': overfitting_start is not None,
                'overfitting_starts_at': overfitting_start,
                'final_generalization_gap': gaps[-1],
                'gap_trend_slope': gap_trend,
                'early_stopping_benefit': validation_scores[-1] - best_val_score
            }
            
        except:
            return {'error': 'Could not analyze early stopping'}
    
    def _find_optimal_n_estimators(self, train_scores, validation_scores):
        """Find optimal number of estimators"""
        try:
            if validation_scores is not None:
                # Use validation scores for optimization
                best_idx = np.argmax(validation_scores)
                return best_idx + 1
            else:
                # Use training scores with diminishing returns analysis
                if len(train_scores) < 10:
                    return len(train_scores)
                
                # Find point of diminishing returns
                improvements = np.diff(train_scores)
                
                # Rolling average of improvements
                window_size = min(5, len(improvements) // 4)
                if window_size > 0:
                    rolling_improvements = np.convolve(
                        improvements, np.ones(window_size)/window_size, mode='valid'
                    )
                    
                    # Find where improvement becomes very small
                    threshold = np.mean(rolling_improvements) * 0.1
                    for i, imp in enumerate(rolling_improvements):
                        if imp < threshold:
                            return i + window_size + 1
                
                return len(train_scores)
                
        except:
            return self.n_estimators
    
    def _analyze_estimator_utilization(self):
        """Analyze how individual estimators are utilized"""
        try:
            # Number of estimators actually used
            n_estimators_used = self.model_.n_estimators_
            
            # Individual tree statistics
            tree_stats = []
            for i, estimator in enumerate(self.model_.estimators_.flat):
                try:
                    tree_depth = estimator.tree_.max_depth
                    n_nodes = estimator.tree_.node_count
                    n_leaves = estimator.tree_.n_leaves
                    
                    tree_stats.append({
                        'depth': tree_depth,
                        'nodes': n_nodes,
                        'leaves': n_leaves
                    })
                except:
                    continue
            
            if tree_stats:
                # Aggregate statistics
                avg_depth = np.mean([ts['depth'] for ts in tree_stats])
                avg_nodes = np.mean([ts['nodes'] for ts in tree_stats])
                avg_leaves = np.mean([ts['leaves'] for ts in tree_stats])
                
                depth_std = np.std([ts['depth'] for ts in tree_stats])
                
                return {
                    'n_estimators_used': n_estimators_used,
                    'avg_tree_depth': avg_depth,
                    'avg_tree_nodes': avg_nodes,
                    'avg_tree_leaves': avg_leaves,
                    'tree_depth_std': depth_std,
                    'tree_complexity_cv': depth_std / (avg_depth + 1e-10),
                    'utilization_rate': n_estimators_used / self.n_estimators
                }
            else:
                return {
                    'n_estimators_used': n_estimators_used,
                    'utilization_rate': n_estimators_used / self.n_estimators
                }
                
        except:
            return {'error': 'Could not analyze estimator utilization'}
    
    def _analyze_learning_trajectory(self):
        """Analyze the learning trajectory of the model"""
        if not self.learning_trajectory_analysis:
            return
        
        try:
            # Get staged predictions for a sample of training data
            sample_size = min(100, len(self.X_original_))
            indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
            X_sample = self.X_original_[indices]
            y_sample = self.y_original_[indices]
            
            # Get staged predictions
            staged_predictions = list(self.model_.staged_predict(X_sample))
            
            # Analyze prediction evolution
            prediction_evolution = self._analyze_prediction_evolution(
                staged_predictions, y_sample
            )
            
            # Learning speed analysis
            learning_speed = self._analyze_learning_speed(staged_predictions, y_sample)
            
            # Convergence characteristics
            convergence_chars = self._analyze_prediction_convergence(staged_predictions)
            
            self.learning_trajectory_analysis_ = {
                'staged_predictions': staged_predictions[:10],  # First 10 for visualization
                'prediction_evolution': prediction_evolution,
                'learning_speed': learning_speed,
                'convergence_characteristics': convergence_chars,
                'sample_size_used': sample_size
            }
            
        except Exception as e:
            self.learning_trajectory_analysis_ = {
                'error': f'Could not analyze learning trajectory: {str(e)}'
            }
    
    def _analyze_prediction_evolution(self, staged_predictions, y_true):
        """Analyze how predictions evolve over boosting iterations"""
        try:
            n_stages = len(staged_predictions)
            errors = []
            
            # Calculate error at each stage
            for pred in staged_predictions:
                mse = mean_squared_error(y_true, pred)
                errors.append(mse)
            
            # Error reduction analysis
            error_reductions = -np.diff(errors)  # Negative because we want reduction
            
            # Learning phases
            early_phase_end = n_stages // 4
            middle_phase_end = 3 * n_stages // 4
            
            early_errors = errors[:early_phase_end] if early_phase_end > 0 else [errors[0]]
            middle_errors = errors[early_phase_end:middle_phase_end] if middle_phase_end > early_phase_end else []
            late_errors = errors[middle_phase_end:] if middle_phase_end < n_stages else []
            
            return {
                'errors_by_stage': errors,
                'error_reductions': error_reductions,
                'initial_error': errors[0],
                'final_error': errors[-1],
                'total_error_reduction': errors[0] - errors[-1],
                'mean_error_reduction': np.mean(error_reductions),
                'early_phase_error_reduction': np.mean(early_errors[0] - early_errors[-1]) if len(early_errors) > 1 else 0,
                'middle_phase_error_reduction': np.mean(middle_errors[0] - middle_errors[-1]) if len(middle_errors) > 1 else 0,
                'late_phase_error_reduction': np.mean(late_errors[0] - late_errors[-1]) if len(late_errors) > 1 else 0,
                'learning_efficiency': (errors[0] - errors[-1]) / n_stages
            }
            
        except:
            return {'error': 'Could not analyze prediction evolution'}
    
    def _analyze_learning_speed(self, staged_predictions, y_true):
        """Analyze the speed of learning"""
        try:
            errors = [mean_squared_error(y_true, pred) for pred in staged_predictions]
            
            # Find iteration where error drops below certain thresholds
            initial_error = errors[0]
            target_50_percent = initial_error * 0.5
            target_75_percent = initial_error * 0.25
            target_90_percent = initial_error * 0.1
            
            iterations_to_50 = None
            iterations_to_75 = None
            iterations_to_90 = None
            
            for i, error in enumerate(errors):
                if iterations_to_50 is None and error <= target_50_percent:
                    iterations_to_50 = i + 1
                if iterations_to_75 is None and error <= target_75_percent:
                    iterations_to_75 = i + 1
                if iterations_to_90 is None and error <= target_90_percent:
                    iterations_to_90 = i + 1
            
            # Learning acceleration/deceleration
            error_reductions = -np.diff(errors)
            
            # First and second derivatives of error reduction
            if len(error_reductions) > 1:
                learning_acceleration = np.diff(error_reductions)
                mean_acceleration = np.mean(learning_acceleration)
            else:
                mean_acceleration = 0
            
            return {
                'iterations_to_50_percent_error': iterations_to_50,
                'iterations_to_75_percent_error': iterations_to_75,
                'iterations_to_90_percent_error': iterations_to_90,
                'initial_learning_rate': error_reductions[0] if len(error_reductions) > 0 else 0,
                'final_learning_rate': error_reductions[-1] if len(error_reductions) > 0 else 0,
                'mean_learning_acceleration': mean_acceleration,
                'learning_speed_category': self._categorize_learning_speed(iterations_to_50, len(errors))
            }
            
        except:
            return {'error': 'Could not analyze learning speed'}
    
    def _categorize_learning_speed(self, iterations_to_50, total_iterations):
        """Categorize learning speed"""
        if iterations_to_50 is None:
            return "Slow - Did not reach 50% error reduction"
        
        ratio = iterations_to_50 / total_iterations
        
        if ratio < 0.1:
            return "Very Fast - Rapid initial learning"
        elif ratio < 0.25:
            return "Fast - Quick convergence"
        elif ratio < 0.5:
            return "Moderate - Steady learning"
        elif ratio < 0.75:
            return "Slow - Gradual improvement"
        else:
            return "Very Slow - Minimal improvement"
    
    def _analyze_prediction_convergence(self, staged_predictions):
        """Analyze convergence of predictions"""
        try:
            n_samples = len(staged_predictions[0])
            n_stages = len(staged_predictions)
            
            # Calculate prediction stability for each sample
            prediction_matrix = np.array(staged_predictions)  # stages x samples
            
            # Stability measures
            final_predictions = prediction_matrix[-1]
            prediction_ranges = np.max(prediction_matrix, axis=0) - np.min(prediction_matrix, axis=0)
            prediction_stds = np.std(prediction_matrix, axis=0)
            
            # Convergence rate for each sample
            convergence_rates = []
            for i in range(n_samples):
                sample_predictions = prediction_matrix[:, i]
                
                # Rate of change in recent iterations
                if n_stages > 10:
                    recent_change = np.std(sample_predictions[-10:])
                    convergence_rates.append(1.0 / (1.0 + recent_change))
                else:
                    convergence_rates.append(0.5)
            
            convergence_rates = np.array(convergence_rates)
            
            return {
                'mean_prediction_range': np.mean(prediction_ranges),
                'mean_prediction_std': np.mean(prediction_stds),
                'mean_convergence_rate': np.mean(convergence_rates),
                'prediction_stability': 1.0 / (1.0 + np.mean(prediction_stds)),
                'samples_converged': np.sum(convergence_rates > 0.8) / n_samples,
                'convergence_uniformity': 1.0 - np.std(convergence_rates)
            }
            
        except:
            return {'error': 'Could not analyze prediction convergence'}
    
    def _analyze_residuals(self):
        """Analyze residuals and their patterns"""
        if not self.residual_analysis:
            return
        
        try:
            # Get predictions and calculate residuals
            y_pred = self.model_.predict(self.X_original_)
            residuals = self.y_original_ - y_pred
            
            # Basic residual statistics
            residual_stats = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'range': np.max(residuals) - np.min(residuals),
                'skewness': self._calculate_skewness(residuals),
                'kurtosis': self._calculate_kurtosis(residuals)
            }
            
            # Residual patterns
            residual_patterns = self._analyze_residual_patterns(residuals, y_pred)
            
            # Staged residual analysis
            staged_residual_analysis = self._analyze_staged_residuals()
            
            # Heteroscedasticity analysis
            heteroscedasticity = self._analyze_heteroscedasticity(residuals, y_pred)
            
            self.residual_analysis_ = {
                'residuals': residuals,
                'residual_statistics': residual_stats,
                'residual_patterns': residual_patterns,
                'staged_residual_analysis': staged_residual_analysis,
                'heteroscedasticity': heteroscedasticity
            }
            
        except Exception as e:
            self.residual_analysis_ = {
                'error': f'Could not analyze residuals: {str(e)}'
            }
    
    def _calculate_skewness(self, values):
        """Calculate skewness of values"""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0
            return np.mean(((values - mean_val) / std_val) ** 3)
        except:
            return 0
    
    def _calculate_kurtosis(self, values):
        """Calculate kurtosis of values"""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0
            return np.mean(((values - mean_val) / std_val) ** 4) - 3
        except:
            return 0
    
    def _analyze_residual_patterns(self, residuals, predictions):
        """Analyze patterns in residuals"""
        try:
            # Autocorrelation of residuals
            autocorr = self._calculate_autocorrelation(residuals)
            
            # Residual vs prediction correlation
            pred_corr = np.corrcoef(residuals, predictions)[0, 1] if len(residuals) > 1 else 0
            
            # Outlier analysis
            residual_threshold = 2 * np.std(residuals)
            outliers = np.abs(residuals) > residual_threshold
            outlier_ratio = np.mean(outliers)
            
            # Distribution analysis
            from scipy import stats
            try:
                # Normality test
                _, normality_p_value = stats.normaltest(residuals)
                is_normal = normality_p_value > 0.05
            except:
                normality_p_value = None
                is_normal = False
            
            return {
                'autocorrelation': autocorr,
                'prediction_correlation': pred_corr,
                'outlier_ratio': outlier_ratio,
                'outlier_threshold': residual_threshold,
                'normality_p_value': normality_p_value,
                'is_normally_distributed': is_normal,
                'pattern_assessment': self._assess_residual_patterns(autocorr, pred_corr, outlier_ratio)
            }
            
        except:
            return {'error': 'Could not analyze residual patterns'}
    
    def _calculate_autocorrelation(self, values, lag=1):
        """Calculate autocorrelation of values"""
        try:
            if len(values) <= lag:
                return 0
            
            v1 = values[:-lag]
            v2 = values[lag:]
            
            return np.corrcoef(v1, v2)[0, 1] if len(v1) > 1 else 0
        except:
            return 0
    
    def _assess_residual_patterns(self, autocorr, pred_corr, outlier_ratio):
        """Assess residual pattern quality"""
        issues = []
        
        if abs(autocorr) > 0.3:
            issues.append("High autocorrelation - possible temporal patterns")
        
        if abs(pred_corr) > 0.2:
            issues.append("Correlation with predictions - possible heteroscedasticity")
        
        if outlier_ratio > 0.1:
            issues.append("High outlier ratio - possible model inadequacy")
        
        if not issues:
            return "Good - Residuals show random pattern"
        else:
            return "Issues detected: " + "; ".join(issues)
    
    def _analyze_staged_residuals(self):
        """Analyze how residuals evolve during boosting"""
        try:
            # Sample data for staged analysis
            sample_size = min(200, len(self.X_original_))
            indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
            X_sample = self.X_original_[indices]
            y_sample = self.y_original_[indices]
            
            # Get staged predictions
            staged_predictions = list(self.model_.staged_predict(X_sample))
            
            # Calculate residuals at different stages
            staged_residuals = []
            residual_stats = []
            
            for pred in staged_predictions[::max(1, len(staged_predictions)//10)]:  # Sample stages
                residuals = y_sample - pred
                staged_residuals.append(residuals)
                
                # Calculate statistics
                stats = {
                    'mean_abs_residual': np.mean(np.abs(residuals)),
                    'std_residual': np.std(residuals),
                    'max_abs_residual': np.max(np.abs(residuals))
                }
                residual_stats.append(stats)
            
            # Analyze residual evolution
            if residual_stats:
                evolution_analysis = {
                    'initial_mean_abs_residual': residual_stats[0]['mean_abs_residual'],
                    'final_mean_abs_residual': residual_stats[-1]['mean_abs_residual'],
                    'residual_reduction': residual_stats[0]['mean_abs_residual'] - residual_stats[-1]['mean_abs_residual'],
                    'residual_reduction_rate': (residual_stats[0]['mean_abs_residual'] - residual_stats[-1]['mean_abs_residual']) / len(residual_stats)
                }
            else:
                evolution_analysis = {}
            
            return {
                'staged_residual_statistics': residual_stats,
                'evolution_analysis': evolution_analysis,
                'stages_analyzed': len(residual_stats)
            }
            
        except:
            return {'error': 'Could not analyze staged residuals'}
    
    def _analyze_heteroscedasticity(self, residuals, predictions):
        """Analyze heteroscedasticity in residuals"""
        try:
            # Breusch-Pagan test approximation
            # Regress squared residuals on predictions
            squared_residuals = residuals ** 2
            
            # Simple correlation test
            bp_correlation = np.corrcoef(squared_residuals, predictions)[0, 1] if len(residuals) > 1 else 0
            
            # Divide predictions into quartiles and compare residual variances
            pred_quartiles = np.percentile(predictions, [25, 50, 75])
            
            q1_mask = predictions <= pred_quartiles[0]
            q2_mask = (predictions > pred_quartiles[0]) & (predictions <= pred_quartiles[1])
            q3_mask = (predictions > pred_quartiles[1]) & (predictions <= pred_quartiles[2])
            q4_mask = predictions > pred_quartiles[2]
            
            q1_var = np.var(residuals[q1_mask]) if np.any(q1_mask) else 0
            q2_var = np.var(residuals[q2_mask]) if np.any(q2_mask) else 0
            q3_var = np.var(residuals[q3_mask]) if np.any(q3_mask) else 0
            q4_var = np.var(residuals[q4_mask]) if np.any(q4_mask) else 0
            
            variance_ratio = max(q1_var, q2_var, q3_var, q4_var) / (min(q1_var, q2_var, q3_var, q4_var) + 1e-10)
            
            # Heteroscedasticity assessment
            if abs(bp_correlation) < 0.1 and variance_ratio < 2.0:
                assessment = "Homoscedastic - Constant variance"
            elif abs(bp_correlation) < 0.2 and variance_ratio < 3.0:
                assessment = "Mild heteroscedasticity"
            else:
                assessment = "Significant heteroscedasticity detected"
            
            return {
                'bp_correlation': bp_correlation,
                'variance_ratio': variance_ratio,
                'quartile_variances': [q1_var, q2_var, q3_var, q4_var],
                'assessment': assessment,
                'is_homoscedastic': abs(bp_correlation) < 0.1 and variance_ratio < 2.0
            }
            
        except:
            return {'error': 'Could not analyze heteroscedasticity'}
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different parameter categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Core Boosting", 
            "🌳 Tree Parameters", 
            "⚙️ Advanced", 
            "📊 Analysis Options",
            "📚 Documentation"
        ])
        
        with tab1:
            st.markdown("**Core Boosting Parameters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.number_input(
                    "Number of Estimators:",
                    value=self.n_estimators,
                    min_value=10,
                    max_value=1000,
                    step=10,
                    help="Number of boosting stages (trees) to fit. More estimators = better performance but slower training",
                    key=f"{key_prefix}_n_estimators"
                )
                
                learning_rate = st.number_input(
                    "Learning Rate:",
                    value=self.learning_rate,
                    min_value=0.001,
                    max_value=1.0,
                    step=0.01,
                    format="%.3f",
                    help="Shrinks contribution of each tree. Lower values need more estimators",
                    key=f"{key_prefix}_learning_rate"
                )
                
                loss = st.selectbox(
                    "Loss Function:",
                    options=['squared_error', 'absolute_error', 'huber', 'quantile'],
                    index=['squared_error', 'absolute_error', 'huber', 'quantile'].index(self.loss),
                    help="Loss function to optimize. Squared error for standard regression, others for robust regression",
                    key=f"{key_prefix}_loss"
                )
            
            with col2:
                subsample = st.number_input(
                    "Subsample Ratio:",
                    value=self.subsample,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="Fraction of samples for fitting individual trees. <1.0 leads to stochastic gradient boosting",
                    key=f"{key_prefix}_subsample"
                )
                
                if loss in ['huber', 'quantile']:
                    alpha = st.number_input(
                        "Alpha Parameter:",
                        value=self.alpha,
                        min_value=0.1,
                        max_value=0.99,
                        step=0.05,
                        help="Alpha parameter for Huber and Quantile loss functions",
                        key=f"{key_prefix}_alpha"
                    )
                else:
                    alpha = self.alpha
                
                random_state = st.number_input(
                    "Random Seed:",
                    value=int(self.random_state),
                    min_value=0,
                    max_value=1000,
                    help="For reproducible results",
                    key=f"{key_prefix}_random_state"
                )
        
        with tab2:
            st.markdown("**Individual Tree Parameters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_depth = st.number_input(
                    "Maximum Tree Depth:",
                    value=self.max_depth,
                    min_value=1,
                    max_value=20,
                    step=1,
                    help="Maximum depth of individual trees. Deeper trees can model more complex patterns",
                    key=f"{key_prefix}_max_depth"
                )
                
                min_samples_split = st.number_input(
                    "Min Samples Split:",
                    value=self.min_samples_split,
                    min_value=2,
                    max_value=50,
                    step=1,# filepath: c:\Users\alise\OneDrive\Desktop\Bachelor Thesis\ml_models\src\ml_plugins\algorithms\gradient_boosting_regressor_plugin.py
                    help="Minimum samples required to split internal node",
                    key=f"{key_prefix}_min_samples_split"
                )
                
                min_samples_leaf = st.number_input(
                    "Min Samples Leaf:",
                    value=self.min_samples_leaf,
                    min_value=1,
                    max_value=20,
                    step=1,
                    help="Minimum samples required at leaf node",
                    key=f"{key_prefix}_min_samples_leaf"
                )
                
                min_weight_fraction_leaf = st.number_input(
                    "Min Weight Fraction Leaf:",
                    value=self.min_weight_fraction_leaf,
                    min_value=0.0,
                    max_value=0.5,
                    step=0.01,
                    help="Minimum weighted fraction of input samples at leaf",
                    key=f"{key_prefix}_min_weight_fraction_leaf"
                )
            
            with col2:
                max_features = st.selectbox(
                    "Max Features per Split:",
                    options=[None, 'sqrt', 'log2', 0.3, 0.5, 0.7],
                    index=[None, 'sqrt', 'log2', 0.3, 0.5, 0.7].index(self.max_features) if self.max_features in [None, 'sqrt', 'log2', 0.3, 0.5, 0.7] else 0,
                    help="Number of features to consider for each split",
                    key=f"{key_prefix}_max_features"
                )
                
                max_leaf_nodes = st.selectbox(
                    "Max Leaf Nodes:",
                    options=[None, 10, 20, 50, 100, 200],
                    index=0 if self.max_leaf_nodes is None else [None, 10, 20, 50, 100, 200].index(self.max_leaf_nodes),
                    help="Maximum number of leaf nodes per tree",
                    key=f"{key_prefix}_max_leaf_nodes"
                )
                
                min_impurity_decrease = st.number_input(
                    "Min Impurity Decrease:",
                    value=self.min_impurity_decrease,
                    min_value=0.0,
                    max_value=0.1,
                    step=0.001,
                    format="%.6f",
                    help="Minimum impurity decrease for split",
                    key=f"{key_prefix}_min_impurity_decrease"
                )
                
                ccp_alpha = st.number_input(
                    "Complexity Parameter (α):",
                    value=self.ccp_alpha,
                    min_value=0.0,
                    max_value=0.1,
                    step=0.001,
                    format="%.6f",
                    help="Cost complexity pruning parameter",
                    key=f"{key_prefix}_ccp_alpha"
                )
        
        with tab3:
            st.markdown("**Advanced Boosting Parameters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                warm_start = st.checkbox(
                    "Warm Start",
                    value=self.warm_start,
                    help="Reuse previous solution to add more estimators incrementally",
                    key=f"{key_prefix}_warm_start"
                )
                
                validation_fraction = st.number_input(
                    "Validation Fraction:",
                    value=self.validation_fraction,
                    min_value=0.1,
                    max_value=0.5,
                    step=0.05,
                    help="Fraction of training data for early stopping validation",
                    key=f"{key_prefix}_validation_fraction"
                )
                
                n_iter_no_change = st.selectbox(
                    "Early Stopping Patience:",
                    options=[None, 5, 10, 15, 20, 30],
                    index=0 if self.n_iter_no_change is None else [None, 5, 10, 15, 20, 30].index(self.n_iter_no_change),
                    help="Number of iterations with no improvement to trigger early stopping",
                    key=f"{key_prefix}_n_iter_no_change"
                )
            
            with col2:
                tol = st.number_input(
                    "Tolerance:",
                    value=self.tol,
                    min_value=1e-6,
                    max_value=1e-2,
                    step=1e-5,
                    format="%.2e",
                    help="Tolerance for early stopping based on loss improvement",
                    key=f"{key_prefix}_tol"
                )
                
                verbose = st.selectbox(
                    "Verbose Output:",
                    options=[0, 1, 2],
                    index=self.verbose,
                    help="Control the verbosity of training output",
                    key=f"{key_prefix}_verbose"
                )
                
                st.markdown("**Boosting Strategy:**")
                st.info("""
                🎯 **Sequential Learning**: Each tree corrects previous errors
                📉 **Gradient Descent**: Optimizes loss function iteratively  
                🔄 **Residual Fitting**: Learns from prediction mistakes
                ⚡ **Adaptive**: Automatically focuses on difficult cases
                """)
        
        with tab4:
            st.markdown("**Analysis and Performance Options**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                compute_feature_importance = st.checkbox(
                    "Feature Importance Analysis",
                    value=self.compute_feature_importance,
                    help="Compute feature importance with boosting evolution",
                    key=f"{key_prefix}_compute_feature_importance"
                )
                
                compute_permutation_importance = st.checkbox(
                    "Permutation Importance",
                    value=self.compute_permutation_importance,
                    help="Compute permutation-based feature importance",
                    key=f"{key_prefix}_compute_permutation_importance"
                )
                
                boosting_analysis = st.checkbox(
                    "Boosting Process Analysis",
                    value=self.boosting_analysis,
                    help="Analyze sequential learning and convergence",
                    key=f"{key_prefix}_boosting_analysis"
                )
                
                learning_trajectory_analysis = st.checkbox(
                    "Learning Trajectory Analysis",
                    value=self.learning_trajectory_analysis,
                    help="Track prediction evolution over iterations",
                    key=f"{key_prefix}_learning_trajectory_analysis"
                )
                
                residual_analysis = st.checkbox(
                    "Residual Analysis",
                    value=self.residual_analysis,
                    help="Comprehensive residual pattern analysis",
                    key=f"{key_prefix}_residual_analysis"
                )
                
                staged_prediction_analysis = st.checkbox(
                    "Staged Prediction Analysis",
                    value=self.staged_prediction_analysis,
                    help="Analyze predictions at each boosting stage",
                    key=f"{key_prefix}_staged_prediction_analysis"
                )
                
                feature_interaction_analysis = st.checkbox(
                    "Feature Interaction Analysis",
                    value=self.feature_interaction_analysis,
                    help="Analyze feature interactions in boosted trees",
                    key=f"{key_prefix}_feature_interaction_analysis"
                )
                
                early_stopping_analysis = st.checkbox(
                    "Early Stopping Analysis",
                    value=self.early_stopping_analysis,
                    help="Analyze optimal stopping points and overfitting",
                    key=f"{key_prefix}_early_stopping_analysis"
                )
            
            with col2:
                overfitting_analysis = st.checkbox(
                    "Overfitting Analysis",
                    value=self.overfitting_analysis,
                    help="Detect and analyze overfitting patterns",
                    key=f"{key_prefix}_overfitting_analysis"
                )
                
                shrinkage_analysis = st.checkbox(
                    "Shrinkage Analysis",
                    value=self.shrinkage_analysis,
                    help="Analyze impact of learning rate on convergence",
                    key=f"{key_prefix}_shrinkage_analysis"
                )
                
                loss_function_analysis = st.checkbox(
                    "Loss Function Analysis",
                    value=self.loss_function_analysis,
                    help="Analyze loss function optimization progress",
                    key=f"{key_prefix}_loss_function_analysis"
                )
                
                ensemble_evolution_analysis = st.checkbox(
                    "Ensemble Evolution Analysis",
                    value=self.ensemble_evolution_analysis,
                    help="Track how ensemble evolves during boosting",
                    key=f"{key_prefix}_ensemble_evolution_analysis"
                )
                
                gradient_analysis = st.checkbox(
                    "Gradient Analysis",
                    value=self.gradient_analysis,
                    help="Analyze gradient patterns and optimization",
                    key=f"{key_prefix}_gradient_analysis"
                )
                
                bias_variance_decomposition = st.checkbox(
                    "Bias-Variance Decomposition",
                    value=self.bias_variance_decomposition,
                    help="Decompose error into bias and variance components",
                    key=f"{key_prefix}_bias_variance_decomposition"
                )
                
                convergence_analysis = st.checkbox(
                    "Convergence Analysis",
                    value=self.convergence_analysis,
                    help="Analyze convergence characteristics and stability",
                    key=f"{key_prefix}_convergence_analysis"
                )
                
                compare_with_random_forest = st.checkbox(
                    "Compare with Random Forest",
                    value=self.compare_with_random_forest,
                    help="Performance comparison with Random Forest",
                    key=f"{key_prefix}_compare_with_random_forest"
                )
            
            # Performance analysis
            st.markdown("**Performance Analysis:**")
            
            learning_curve_analysis = st.checkbox(
                "Learning Curve Analysis",
                value=self.learning_curve_analysis,
                help="Analyze learning curves for training data efficiency",
                key=f"{key_prefix}_learning_curve_analysis"
            )
            
            hyperparameter_sensitivity_analysis = st.checkbox(
                "Hyperparameter Sensitivity",
                value=self.hyperparameter_sensitivity_analysis,
                help="Analyze sensitivity to hyperparameter changes",
                key=f"{key_prefix}_hyperparameter_sensitivity_analysis"
            )
            
            computational_efficiency_analysis = st.checkbox(
                "Computational Efficiency Analysis",
                value=self.computational_efficiency_analysis,
                help="Analyze training and prediction speed",
                key=f"{key_prefix}_computational_efficiency_analysis"
            )
            
            cross_validation_analysis = st.checkbox(
                "Cross-Validation Analysis",
                value=self.cross_validation_analysis,
                help="Comprehensive cross-validation assessment",
                key=f"{key_prefix}_cross_validation_analysis"
            )
            
            performance_benchmarking = st.checkbox(
                "Performance Benchmarking",
                value=self.performance_benchmarking,
                help="Compare against baseline models",
                key=f"{key_prefix}_performance_benchmarking"
            )
            
            # Cross-validation settings
            if cross_validation_analysis:
                st.markdown("**Cross-Validation Settings:**")
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
            
            # Visualization options
            st.markdown("**Visualization Options:**")
            visualize_trees = st.checkbox(
                "Visualize Individual Trees",
                value=self.visualize_trees,
                help="Visualize structure of individual trees (for small ensembles)",
                key=f"{key_prefix}_visualize_trees"
            )
            
            if visualize_trees:
                max_trees_to_visualize = st.number_input(
                    "Max Trees to Visualize:",
                    value=self.max_trees_to_visualize,
                    min_value=1,
                    max_value=5,
                    step=1,
                    help="Maximum number of trees to visualize",
                    key=f"{key_prefix}_max_trees_to_visualize"
                )
            else:
                max_trees_to_visualize = self.max_trees_to_visualize
        
        with tab5:
            st.markdown("**Gradient Boosting Regressor - Sequential Boosting**")
            
            # Algorithm information
            if st.button("📚 Algorithm Information", key=f"{key_prefix}_algo_info"):
                st.markdown("""
                **Gradient Boosting - Sequential Ensemble Learning**
                
                Gradient Boosting builds models sequentially, where each subsequent model learns
                from the mistakes of the previous ones by fitting to the residuals. It combines
                weak learners (typically decision trees) to create a strong predictive model.
                
                **Core Principles:**
                • **Sequential Learning** - Models built one after another
                • **Residual Fitting** - Each model corrects previous errors
                • **Gradient Descent** - Optimizes loss function iteratively
                • **Weak Learner Combination** - Combines many simple models
                • **Additive Model** - F(x) = Σ γ_m * h_m(x)
                • **Shrinkage** - Learning rate controls contribution of each model
                
                **Key Advantages:**
                • 🎯 **High Accuracy** - Often achieves excellent predictive performance
                • 🔄 **Adaptive Learning** - Focuses on difficult cases automatically
                • 📊 **Flexible Loss Functions** - Supports various regression objectives
                • 🛡️ **Built-in Regularization** - Multiple overfitting prevention mechanisms
                • 📈 **Feature Importance** - Provides meaningful feature rankings
                • ⚡ **Early Stopping** - Automatic convergence detection
                """)
            
            # When to use Gradient Boosting
            if st.button("🎯 When to Use Gradient Boosting", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • High accuracy requirements
                • Complex non-linear relationships
                • Moderate to large datasets
                • Feature interactions important
                • Regression with various loss functions needed
                
                **Data Characteristics:**
                • Tabular data with mixed feature types
                • Presence of noise but not extreme outliers
                • Sufficient training data (hundreds to thousands of samples)
                • Features with varying importance levels
                • Non-linear target relationships
                
                **Examples:**
                • Predictive modeling competitions (Kaggle)
                • Financial risk modeling and pricing
                • Marketing response prediction
                • Quality control and process optimization
                • Demand forecasting and time series
                • Medical diagnosis and prognosis
                • Real estate price prediction
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Excellent predictive accuracy on tabular data
                ✅ Handles mixed data types naturally
                ✅ Built-in feature selection through importance
                ✅ Robust to outliers (with appropriate loss functions)
                ✅ No need for extensive data preprocessing
                ✅ Provides feature importance rankings
                ✅ Early stopping prevents overfitting
                ✅ Flexible loss functions for different objectives
                ✅ Handles missing values reasonably well
                ✅ Good performance with default parameters
                
                **Limitations:**
                ❌ Sequential training (not parallelizable across trees)
                ❌ Sensitive to hyperparameters (learning rate, depth)
                ❌ Can overfit with too many iterations
                ❌ Requires careful tuning for optimal performance
                ❌ Less interpretable than single trees
                ❌ Slower training than Random Forest
                ❌ Memory intensive for large ensembles
                ❌ Can struggle with very sparse data
                """)
            
            # Gradient Boosting vs other methods
            if st.button("🔍 Gradient Boosting vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **Gradient Boosting vs Other Ensemble Methods:**
                
                **Gradient Boosting vs Random Forest:**
                • GB: Sequential learning, higher accuracy potential
                • RF: Parallel training, faster, more robust to overfitting
                • GB: Better for complex patterns, requires more tuning
                • RF: Better for quick deployment, less sensitive to parameters
                
                **Gradient Boosting vs AdaBoost:**
                • GB: Gradient-based optimization, more flexible loss functions
                • Ada: Weight-based boosting, binary classification focus
                • GB: Better for regression, handles noise better
                • Ada: Simpler concept, faster for simple problems
                
                **Gradient Boosting vs XGBoost/LightGBM:**
                • GB: Standard implementation, good baseline
                • XGB/LGB: Optimized implementations, faster, better accuracy
                • GB: More interpretable training process
                • XGB/LGB: Better for competitions, production systems
                
                **Gradient Boosting vs Neural Networks:**
                • GB: Better for tabular data, easier to tune
                • NN: Better for unstructured data, more flexible
                • GB: Requires less data, faster training on small datasets
                • NN: Better scalability, can model any function
                
                **Gradient Boosting vs Linear Models:**
                • GB: Handles non-linearity and interactions automatically
                • Linear: More interpretable, faster prediction
                • GB: Better accuracy on complex patterns
                • Linear: Better extrapolation, less overfitting risk
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Gradient Boosting Best Practices:**
                
                **Hyperparameter Tuning Priority:**
                1. **learning_rate**: Start with 0.1, lower for more estimators
                2. **n_estimators**: Start with 100, increase with lower learning rate
                3. **max_depth**: Start with 3-6, deeper for complex patterns
                4. **subsample**: Use 0.8-1.0, lower values for regularization
                
                **Training Strategy:**
                1. Use **early stopping** with validation set
                2. Monitor training vs validation loss curves
                3. Start with conservative parameters, then tune
                4. Use **cross-validation** for robust performance assessment
                
                **Overfitting Prevention:**
                1. Use validation sets and early stopping
                2. Lower learning rate with more estimators
                3. Reduce max_depth and increase min_samples_split
                4. Use subsample < 1.0 for stochastic boosting
                5. Apply regularization through ccp_alpha
                
                **Performance Optimization:**
                1. **Learning Rate Schedule**: Start high, reduce over time
                2. **Staged Predictions**: Monitor improvement at each stage
                3. **Feature Engineering**: Good features improve boosting
                4. **Loss Function**: Choose appropriate for your problem
                
                **Model Validation:**
                1. Plot learning curves to detect overfitting
                2. Analyze residuals for patterns
                3. Check feature importance stability
                4. Compare with simpler baselines
                
                **Production Considerations:**
                1. Save optimal n_estimators from validation
                2. Monitor prediction time vs accuracy tradeoff
                3. Consider ensemble pruning for deployment
                4. Regular retraining for data drift
                """)
            
            # Advanced techniques
            if st.button("🚀 Advanced Techniques", key=f"{key_prefix}_advanced"):
                st.markdown("""
                **Advanced Gradient Boosting Techniques:**
                
                **Loss Function Customization:**
                • **Custom Loss Functions**: Implement domain-specific objectives
                • **Multi-objective Boosting**: Optimize multiple targets simultaneously
                • **Quantile Regression**: Predict confidence intervals
                • **Robust Loss Functions**: Handle outliers better
                
                **Regularization Strategies:**
                • **Dropout Boosting**: Randomly skip trees during training
                • **Shrinkage Schedules**: Adaptive learning rate reduction
                • **Feature Subsampling**: Random feature selection per tree
                • **Early Stopping**: Validation-based convergence detection
                
                **Ensemble Enhancement:**
                • **Gradient Boosting Variants**: MART, GBRT, functional gradient descent
                • **Multi-class Extensions**: One-vs-rest, softmax boosting
                • **Online Boosting**: Incremental learning for streaming data
                • **Parallel Boosting Approximations**: Speed up training
                
                **Feature Engineering:**
                • **Interaction Detection**: Use boosting to find feature interactions
                • **Automatic Feature Selection**: Importance-based filtering
                • **Feature Transformation**: Non-linear feature mapping
                • **Staged Feature Importance**: Track importance evolution
                
                **Model Interpretation:**
                • **Partial Dependence Plots**: Understand feature effects
                • **SHAP Values**: Individual prediction explanations
                • **Tree Ensemble Simplification**: Extract rule sets
                • **Gradient Analysis**: Understand optimization path
                
                **Hyperparameter Optimization:**
                • **Bayesian Optimization**: Efficient parameter search
                • **Multi-fidelity Optimization**: Use learning curves for early stopping
                • **Population-based Training**: Evolve hyperparameters during training
                • **AutoML Integration**: Automated boosting configuration
                """)
        
        return {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "loss": loss,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "subsample": subsample,
            "alpha": alpha,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
            "validation_fraction": validation_fraction,
            "n_iter_no_change": n_iter_no_change,
            "tol": tol,
            "ccp_alpha": ccp_alpha,
            "compute_feature_importance": compute_feature_importance,
            "compute_permutation_importance": compute_permutation_importance,
            "boosting_analysis": boosting_analysis,
            "learning_trajectory_analysis": learning_trajectory_analysis,
            "residual_analysis": residual_analysis,
            "staged_prediction_analysis": staged_prediction_analysis,
            "feature_interaction_analysis": feature_interaction_analysis,
            "learning_curve_analysis": learning_curve_analysis,
            "hyperparameter_sensitivity_analysis": hyperparameter_sensitivity_analysis,
            "computational_efficiency_analysis": computational_efficiency_analysis,
            "early_stopping_analysis": early_stopping_analysis,
            "overfitting_analysis": overfitting_analysis,
            "shrinkage_analysis": shrinkage_analysis,
            "loss_function_analysis": loss_function_analysis,
            "compare_with_random_forest": compare_with_random_forest,
            "compare_with_linear_model": True,
            "ensemble_evolution_analysis": ensemble_evolution_analysis,
            "visualize_trees": visualize_trees,
            "max_trees_to_visualize": max_trees_to_visualize,
            "feature_importance_analysis": True,
            "prediction_distribution_analysis": True,
            "cross_validation_analysis": cross_validation_analysis,
            "cv_folds": cv_folds,
            "performance_benchmarking": performance_benchmarking,
            "gradient_analysis": gradient_analysis,
            "bias_variance_decomposition": bias_variance_decomposition,
            "convergence_analysis": convergence_analysis
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return GradientBoostingRegressorPlugin(
            n_estimators=hyperparameters.get("n_estimators", self.n_estimators),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            loss=hyperparameters.get("loss", self.loss),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            min_samples_split=hyperparameters.get("min_samples_split", self.min_samples_split),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", self.min_samples_leaf),
            min_weight_fraction_leaf=hyperparameters.get("min_weight_fraction_leaf", self.min_weight_fraction_leaf),
            max_features=hyperparameters.get("max_features", self.max_features),
            max_leaf_nodes=hyperparameters.get("max_leaf_nodes", self.max_leaf_nodes),
            min_impurity_decrease=hyperparameters.get("min_impurity_decrease", self.min_impurity_decrease),
            subsample=hyperparameters.get("subsample", self.subsample),
            alpha=hyperparameters.get("alpha", self.alpha),
            random_state=hyperparameters.get("random_state", self.random_state),
            verbose=hyperparameters.get("verbose", self.verbose),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            validation_fraction=hyperparameters.get("validation_fraction", self.validation_fraction),
            n_iter_no_change=hyperparameters.get("n_iter_no_change", self.n_iter_no_change),
            tol=hyperparameters.get("tol", self.tol),
            ccp_alpha=hyperparameters.get("ccp_alpha", self.ccp_alpha),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            compute_permutation_importance=hyperparameters.get("compute_permutation_importance", self.compute_permutation_importance),
            boosting_analysis=hyperparameters.get("boosting_analysis", self.boosting_analysis),
            learning_trajectory_analysis=hyperparameters.get("learning_trajectory_analysis", self.learning_trajectory_analysis),
            residual_analysis=hyperparameters.get("residual_analysis", self.residual_analysis),
            staged_prediction_analysis=hyperparameters.get("staged_prediction_analysis", self.staged_prediction_analysis),
            feature_interaction_analysis=hyperparameters.get("feature_interaction_analysis", self.feature_interaction_analysis),
            learning_curve_analysis=hyperparameters.get("learning_curve_analysis", self.learning_curve_analysis),
            hyperparameter_sensitivity_analysis=hyperparameters.get("hyperparameter_sensitivity_analysis", self.hyperparameter_sensitivity_analysis),
            computational_efficiency_analysis=hyperparameters.get("computational_efficiency_analysis", self.computational_efficiency_analysis),
            early_stopping_analysis=hyperparameters.get("early_stopping_analysis", self.early_stopping_analysis),
            overfitting_analysis=hyperparameters.get("overfitting_analysis", self.overfitting_analysis),
            shrinkage_analysis=hyperparameters.get("shrinkage_analysis", self.shrinkage_analysis),
            loss_function_analysis=hyperparameters.get("loss_function_analysis", self.loss_function_analysis),
            compare_with_random_forest=hyperparameters.get("compare_with_random_forest", self.compare_with_random_forest),
            compare_with_linear_model=hyperparameters.get("compare_with_linear_model", True),
            ensemble_evolution_analysis=hyperparameters.get("ensemble_evolution_analysis", self.ensemble_evolution_analysis),
            visualize_trees=hyperparameters.get("visualize_trees", self.visualize_trees),
            max_trees_to_visualize=hyperparameters.get("max_trees_to_visualize", self.max_trees_to_visualize),
            feature_importance_analysis=hyperparameters.get("feature_importance_analysis", self.feature_importance_analysis),
            prediction_distribution_analysis=hyperparameters.get("prediction_distribution_analysis", self.prediction_distribution_analysis),
            cross_validation_analysis=hyperparameters.get("cross_validation_analysis", self.cross_validation_analysis),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            performance_benchmarking=hyperparameters.get("performance_benchmarking", self.performance_benchmarking),
            gradient_analysis=hyperparameters.get("gradient_analysis", self.gradient_analysis),
            bias_variance_decomposition=hyperparameters.get("bias_variance_decomposition", self.bias_variance_decomposition),
            convergence_analysis=hyperparameters.get("convergence_analysis", self.convergence_analysis)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Gradient Boosting (minimal preprocessing needed)"""
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
        """Check if Gradient Boosting is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Gradient Boosting requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Gradient Boosting Regressor requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # Sample size assessment for sequential boosting
            if n_samples >= 1000:
                advantages.append(f"Large sample size ({n_samples}) - excellent for sequential boosting")
            elif n_samples >= 200:
                advantages.append(f"Good sample size ({n_samples}) - adequate for gradient boosting")
            else:
                considerations.append(f"Small sample size ({n_samples}) - watch for overfitting")
            
            # Feature dimensionality assessment
            if n_features >= 50:
                advantages.append(f"High dimensionality ({n_features}) - good for complex pattern detection")
            elif n_features >= 10:
                advantages.append(f"Moderate dimensionality ({n_features}) - suitable for boosting")
            else:
                considerations.append(f"Low dimensionality ({n_features}) - may benefit from feature engineering")
            
            # Data characteristics favorable to Gradient Boosting
            advantages.append("Sequential learning excels at complex non-linear patterns")
            advantages.append("Built-in feature selection through iterative importance")
            advantages.append("Robust to outliers with appropriate loss functions")
            advantages.append("Excellent for predictive accuracy on tabular data")
            
            # Check feature-to-sample ratio
            feature_sample_ratio = n_features / n_samples
            if feature_sample_ratio > 0.8:
                considerations.append(f"High feature-to-sample ratio ({feature_sample_ratio:.2f}) - risk of overfitting")
            elif feature_sample_ratio > 0.3:
                considerations.append(f"Moderate feature-to-sample ratio ({feature_sample_ratio:.2f}) - use regularization")
            else:
                advantages.append(f"Good feature-to-sample ratio ({feature_sample_ratio:.2f}) - ideal for boosting")
            
            # Sequential learning advantages
            advantages.append("Sequential error correction focuses on difficult cases")
            advantages.append("Multiple loss functions available for robust regression")
            
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
                f"🎯 Suitability for Gradient Boosting: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🚀 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance with Gradient Boosting specific insights"""
        if not self.is_fitted_:
            return None
        
        if not self.feature_importance_analysis_:
            return None
        
        analysis = self.feature_importance_analysis_
        
        if 'error' in analysis:
            return {'error': analysis['error']}
        
        # Extract importance information
        builtin_importance = analysis['builtin_importance']
        permutation_importance = analysis.get('permutation_importance')
        feature_ranking = analysis['feature_ranking']
        
        # Create feature importance dictionary with Gradient Boosting specific metrics
        feature_importance = {}
        for i, name in enumerate(self.feature_names_):
            feature_importance[name] = {
                'gini_importance': builtin_importance[i],
                'permutation_importance': permutation_importance[i] if permutation_importance is not None else None,
                'rank': np.where(feature_ranking == i)[0][0] + 1,
                'evolution_stability': analysis['importance_evolution'].get('feature_stability', {}).get(i, {}).get('stability_score', 0.0),
                'boosting_selection_score': analysis['boosting_feature_selection'].get('selection_scores', [0.0] * len(self.feature_names_))[i],
                'is_boosting_selected': analysis['boosting_feature_selection'].get('recommended_features', [False] * len(self.feature_names_))[i]
            }
        
        # Get top features
        top_features = analysis['top_features']
        
        return {
            'feature_importance': feature_importance,
            'top_features': [(f['name'], f['builtin_importance'], f.get('permutation_importance')) for f in top_features],
            'importance_statistics': analysis['importance_statistics'],
            'boosting_insights': {
                'importance_evolution': analysis.get('importance_evolution', {}),
                'feature_selection': analysis.get('boosting_feature_selection', {}),
                'staged_importance': analysis.get('staged_importance', {})
            },
            'ensemble_info': {
                'n_estimators': self.model_.n_estimators_,
                'boosting_stability': analysis['importance_statistics'].get('boosting_stability', 0.0),
                'selected_features_count': np.sum(analysis['boosting_feature_selection'].get('recommended_features', []))
            },
            'interpretation': 'Gradient Boosting feature importance with sequential learning analysis'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "algorithm": "Gradient Boosting Regressor",
            "boosting_structure": {
                "n_estimators": self.model_.n_estimators_,
                "learning_rate": self.learning_rate,
                "loss_function": self.loss,
                "sequential_learning": True
            },
            "tree_parameters": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "subsample": self.subsample
            },
            "boosting_analysis": {
                "training_progression": self.boosting_analysis_.get('progression_analysis', {}),
                "convergence": self.boosting_analysis_.get('convergence_analysis', {}),
                "early_stopping": self.early_stopping_analysis_.get('early_stopping_analysis', {})
            },
            "interpretability": {
                "feature_importance_available": True,
                "staged_predictions": True,
                "residual_analysis": True,
                "learning_trajectory": True
            }
        }
    
    def get_boosting_analysis(self) -> Dict[str, Any]:
        """Get comprehensive boosting analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        return {
            "feature_importance_analysis": self.feature_importance_analysis_,
            "boosting_analysis": self.boosting_analysis_,
            "learning_trajectory_analysis": self.learning_trajectory_analysis_,
            "residual_analysis": self.residual_analysis_,
            "staged_prediction_analysis": self.staged_prediction_analysis_,
            "feature_interaction_analysis": self.feature_interaction_analysis_,
            "learning_curve_analysis": self.learning_curve_analysis_,
            "hyperparameter_sensitivity_analysis": self.hyperparameter_sensitivity_analysis_,
            "computational_efficiency_analysis": self.computational_efficiency_analysis_,
            "early_stopping_analysis": self.early_stopping_analysis_,
            "overfitting_analysis": self.overfitting_analysis_,
            "shrinkage_analysis": self.shrinkage_analysis_,
            "loss_function_analysis": self.loss_function_analysis_,
            "random_forest_comparison": self.random_forest_comparison_,
            "linear_model_comparison": self.linear_model_comparison_,
            "ensemble_evolution_analysis": self.ensemble_evolution_analysis_,
            "cross_validation_analysis": self.cross_validation_analysis_,
            "performance_benchmarking": self.performance_benchmarking_,
            "gradient_analysis": self.gradient_analysis_,
            "bias_variance_decomposition": self.bias_variance_decomposition_,
            "convergence_analysis": self.convergence_analysis_
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Gradient Boosting Regressor",
            "type": "Sequential ensemble with iterative residual learning",
            "training_completed": True,
            "boosting_characteristics": {
                "sequential_learning": True,
                "residual_fitting": True,
                "gradient_optimization": True,
                "adaptive_weighting": False,
                "iterative_improvement": True
            },
            "ensemble_structure": {
                "n_estimators": self.model_.n_estimators_,
                "learning_rate": self.learning_rate,
                "loss_function": self.loss,
                "subsample_ratio": self.subsample
            },
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "loss": self.loss,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "subsample": self.subsample
            }
        }
        
        # Add boosting analysis information if available
        if self.boosting_analysis_:
            info["boosting_progress"] = {
                "convergence_status": self.boosting_analysis_.get('convergence_analysis', {}).get('converged', False),
                "overfitting_detected": self.boosting_analysis_.get('convergence_analysis', {}).get('overfitting_detected', False),
                "optimal_n_estimators": self.boosting_analysis_.get('convergence_analysis', {}).get('optimal_n_estimators', self.n_estimators),
                "final_training_score": self.boosting_analysis_.get('final_train_score')
            }
        
        if self.early_stopping_analysis_:
            info["early_stopping"] = {
                "early_stopping_used": self.n_iter_no_change is not None,
                "stopping_analysis": self.early_stopping_analysis_.get('early_stopping_analysis', {})
            }
        
        return info

    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Gradient Boosting Regressor-specific metrics based on the fitted model's
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
            A dictionary of Gradient Boosting Regressor-specific metrics.
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
            if isinstance(current, (list, np.ndarray)) and len(current) > 10 and not path.endswith("_scores"): # allow if path indicates multiple scores
                 return default
            return current if pd.notna(current) else default

        # --- Feature Importance Analysis ---
        if hasattr(self, 'feature_importance_analysis_') and self.feature_importance_analysis_ and not self.feature_importance_analysis_.get('error'):
            fi_analysis = self.feature_importance_analysis_
            metrics['fi_mean_importance'] = safe_get(fi_analysis, 'importance_statistics.mean_importance')
            metrics['fi_gini_coefficient'] = safe_get(fi_analysis, 'importance_statistics.gini_coefficient')
            metrics['fi_boosting_stability'] = safe_get(fi_analysis, 'importance_statistics.boosting_stability')
            metrics['fi_recommended_features_count'] = safe_get(fi_analysis, 'boosting_feature_selection.n_recommended')
            metrics['fi_overall_importance_stability'] = safe_get(fi_analysis, 'importance_evolution.overall_stability')


        # --- Boosting Process Analysis ---
        if hasattr(self, 'boosting_analysis_') and self.boosting_analysis_ and not self.boosting_analysis_.get('error'):
            boost_an = self.boosting_analysis_
            metrics['boost_n_estimators_used'] = safe_get(boost_an, 'n_estimators_used')
            metrics['boost_final_train_score'] = safe_get(boost_an, 'final_train_score')
            metrics['boost_prog_total_improvement'] = safe_get(boost_an, 'progression_analysis.total_improvement')
            metrics['boost_prog_learning_efficiency'] = safe_get(boost_an, 'progression_analysis.learning_efficiency')
            metrics['boost_prog_final_val_score'] = safe_get(boost_an, 'progression_analysis.validation_analysis.final_val_score')
            metrics['boost_prog_generalization_gap'] = safe_get(boost_an, 'progression_analysis.validation_analysis.generalization_gap')
            metrics['boost_seq_improvement_efficiency'] = safe_get(boost_an, 'improvement_analysis.sequential_efficiency')
            metrics['boost_seq_diminishing_returns_ratio'] = safe_get(boost_an, 'improvement_analysis.diminishing_returns_ratio')
            metrics['boost_train_converged'] = safe_get(boost_an, 'convergence_analysis.train_convergence.converged')
            metrics['boost_optimal_n_estimators'] = safe_get(boost_an, 'convergence_analysis.optimal_n_estimators')
            metrics['boost_overfitting_detected_internal'] = safe_get(boost_an, 'convergence_analysis.overfitting_detected')

        # --- Learning Trajectory Analysis ---
        if hasattr(self, 'learning_trajectory_analysis_') and self.learning_trajectory_analysis_ and not self.learning_trajectory_analysis_.get('error'):
            lt_an = self.learning_trajectory_analysis_
            metrics['lt_final_sample_error_mse'] = safe_get(lt_an, 'prediction_evolution.final_error')
            metrics['lt_pred_evol_learning_efficiency'] = safe_get(lt_an, 'prediction_evolution.learning_efficiency')
            metrics['lt_iters_to_50_pct_error'] = safe_get(lt_an, 'learning_speed.iterations_to_50_percent_error')
            metrics['lt_prediction_stability'] = safe_get(lt_an, 'convergence_characteristics.prediction_stability')
            metrics['lt_mean_convergence_rate'] = safe_get(lt_an, 'convergence_characteristics.mean_convergence_rate')


        # --- Residual Analysis ---
        if hasattr(self, 'residual_analysis_') and self.residual_analysis_ and not self.residual_analysis_.get('error'):
            res_an = self.residual_analysis_
            metrics['res_mean'] = safe_get(res_an, 'residual_statistics.mean')
            metrics['res_std'] = safe_get(res_an, 'residual_statistics.std')
            metrics['res_skewness'] = safe_get(res_an, 'residual_statistics.skewness')
            metrics['res_kurtosis'] = safe_get(res_an, 'residual_statistics.kurtosis')
            metrics['res_pattern_autocorrelation'] = safe_get(res_an, 'residual_patterns.autocorrelation')
            metrics['res_pattern_normality_p_value'] = safe_get(res_an, 'residual_patterns.normality_p_value')
            metrics['res_pattern_is_normal'] = safe_get(res_an, 'residual_patterns.is_normally_distributed')
            metrics['res_hetero_bp_correlation'] = safe_get(res_an, 'heteroscedasticity.bp_correlation')
            metrics['res_hetero_variance_ratio'] = safe_get(res_an, 'heteroscedasticity.variance_ratio')
            metrics['res_hetero_is_homoscedastic'] = safe_get(res_an, 'heteroscedasticity.is_homoscedastic')

        # --- Early Stopping Analysis ---
        if hasattr(self, 'early_stopping_analysis_') and self.early_stopping_analysis_ and not self.early_stopping_analysis_.get('error'):
            es_an = self.early_stopping_analysis_ # This is a direct attribute from __init__
            # The detailed early stopping logic is often within _analyze_boosting_convergence or _analyze_early_stopping_point
            # and results might be in boosting_analysis_ or a dedicated early_stopping_analysis_ if populated by _analyze_early_stopping
            # For now, let's assume some key metrics might be directly in self.early_stopping_analysis_ if it's populated by its dedicated method
            metrics['es_optimal_n_estimators'] = safe_get(es_an, 'early_stopping_analysis.optimal_n_estimators') # Path based on example in boosting_analysis
            metrics['es_best_validation_score'] = safe_get(es_an, 'early_stopping_analysis.best_validation_score')
            metrics['es_overfitting_detected'] = safe_get(es_an, 'early_stopping_analysis.overfitting_detected')
            metrics['es_benefit'] = safe_get(es_an, 'early_stopping_analysis.early_stopping_benefit')


        # --- Convergence Analysis (Top Level) ---
        if hasattr(self, 'convergence_analysis_') and self.convergence_analysis_ and not self.convergence_analysis_.get('error'):
            conv_an = self.convergence_analysis_ # This is a direct attribute
            # This might be a summary if _analyze_convergence populates it differently from boosting_analysis.convergence_analysis
            metrics['conv_overall_converged'] = safe_get(conv_an, 'converged') # Assuming a simple structure
            metrics['conv_stability_score'] = safe_get(conv_an, 'stability_score')


        # --- Bias-Variance Decomposition ---
        if hasattr(self, 'bias_variance_decomposition_') and self.bias_variance_decomposition_ and not self.bias_variance_decomposition_.get('error'):
            bv_an = self.bias_variance_decomposition_
            metrics['bv_bias_squared'] = safe_get(bv_an, 'bias_variance.bias_squared') # Assuming structure like {'bias_variance': {'bias_squared': ...}}
            metrics['bv_variance'] = safe_get(bv_an, 'bias_variance.variance')
            metrics['bv_error'] = safe_get(bv_an, 'bias_variance.error')
            metrics['bv_bias_variance_ratio'] = safe_get(bv_an, 'assessment.bias_variance_ratio')
            metrics['bv_balance_score'] = safe_get(bv_an, 'assessment.balance_score')

        # --- Computational Efficiency ---
        if hasattr(self, 'computational_efficiency_analysis_') and self.computational_efficiency_analysis_ and not self.computational_efficiency_analysis_.get('error'):
            comp_eff = self.computational_efficiency_analysis_
            metrics['comp_training_time'] = safe_get(comp_eff, 'training_time.total_time')
            metrics['comp_prediction_time_per_1k'] = safe_get(comp_eff, 'prediction_speed.time_per_1000_samples')

        # --- Cross-Validation Analysis (Summary) ---
        if hasattr(self, 'cross_validation_analysis_') and self.cross_validation_analysis_ and not self.cross_validation_analysis_.get('error'):
            cv_an = self.cross_validation_analysis_
            metrics['cv_mean_r2'] = safe_get(cv_an, 'cv_scores.r2.mean')
            metrics['cv_std_r2'] = safe_get(cv_an, 'cv_scores.r2.std')
            metrics['cv_mean_neg_mse'] = safe_get(cv_an, 'cv_scores.neg_mean_squared_error.mean') # or other primary metric
            metrics['cv_generalization_gap'] = safe_get(cv_an, 'performance_assessment.generalization_gap')

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
    return GradientBoostingRegressorPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Gradient Boosting Regressor Plugin
    """
    print("Testing Gradient Boosting Regressor Plugin...")
    
    try:
        # Create sample data
        np.random.seed(42)
        
        # Generate synthetic regression data
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            noise=0.1,
            random_state=42
        )
        
        # Add some non-linear patterns (good for boosting)
        y = y + 0.1 * X[:, 0] * X[:, 1] + 0.05 * X[:, 2]**2 + 0.02 * X[:, 3]**3
        
        print(f"\n📊 Test Dataset:")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"Target variance: {np.var(y):.3f}")
        print(f"Non-linear patterns added for boosting effectiveness")
        
        # Test Gradient Boosting
        print(f"\n🚀 Testing Gradient Boosting Regression...")
        
        # Create DataFrame for proper feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        plugin = GradientBoostingRegressorPlugin(
            n_estimators=100,
            learning_rate=0.1,
            loss='squared_error',
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.9,
            compute_feature_importance=True,
            compute_permutation_importance=True,
            boosting_analysis=True,
            learning_trajectory_analysis=True,
            residual_analysis=True,
            staged_prediction_analysis=True,
            early_stopping_analysis=True,
            overfitting_analysis=True,
            compare_with_random_forest=True,
            cross_validation_analysis=True,
            random_state=42
        )
        
        # Check compatibility
        compatible, message = plugin.is_compatible_with_data(X_df, y)
        print(f"✅ Compatibility: {message}")
        
        if compatible:
            # Train model
            plugin.fit(X_df, y)
            
            # Make predictions
            y_pred = plugin.predict(X_df)
            
            # Test uncertainty prediction
            uncertainty_results = plugin.predict_with_uncertainty(X_df[:100])
            
            # Test staged predictions
            staged_preds = plugin.predict_staged(X_df[:10], n_estimators=10)
            
            # Evaluate
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            print(f"\n📊 Gradient Boosting Results:")
            print(f"Training R²: {r2:.4f}")
            print(f"Training MSE: {mse:.4f}")
            print(f"Estimators used: {plugin.model_.n_estimators_}")
            
            # Boosting analysis
            if plugin.boosting_analysis_:
                boosting = plugin.boosting_analysis_
                if 'progression_analysis' in boosting:
                    prog = boosting['progression_analysis']
                    print(f"Total improvement: {prog.get('total_improvement', 0):.4f}")
                    print(f"Learning efficiency: {prog.get('learning_efficiency', 0):.3f}")
                
                if 'convergence_analysis' in boosting:
                    conv = boosting['convergence_analysis']
                    print(f"Converged: {conv.get('converged', False)}")
                    print(f"Optimal estimators: {conv.get('optimal_n_estimators', 'Unknown')}")
            
            # Learning trajectory
            if plugin.learning_trajectory_analysis_:
                trajectory = plugin.learning_trajectory_analysis_
                if 'learning_speed' in trajectory:
                    speed = trajectory['learning_speed']
                    print(f"Learning speed: {speed.get('learning_speed_category', 'Unknown')}")
            
            # Feature importance
            feature_imp = plugin.get_feature_importance()
            if feature_imp and 'top_features' in feature_imp:
                print(f"\nTop 5 Features:")
                for i, (name, importance, _) in enumerate(feature_imp['top_features'][:5]):
                    evolution_stability = feature_imp['feature_importance'][name]['evolution_stability']
                    print(f"{i+1}. {name}: {importance:.4f} (stability: {evolution_stability:.3f})")
            
            # Model parameters
            model_params = plugin.get_model_params()
            print(f"\nModel Structure:")
            print(f"Estimators: {model_params['boosting_structure']['n_estimators']}")
            print(f"Learning Rate: {model_params['boosting_structure']['learning_rate']}")
            print(f"Loss Function: {model_params['boosting_structure']['loss_function']}")
            
            # Uncertainty analysis
            print(f"\nUncertainty Analysis (first 5 samples):")
            uncertainty = uncertainty_results
            for i in range(min(5, len(uncertainty['predictions']))):
                pred = uncertainty['predictions'][i]
                std = uncertainty['prediction_std'][i]
                stability = uncertainty['boosting_stability'][i]
                print(f"Sample {i+1}: {pred:.3f} ± {std:.3f} (stability: {stability:.3f})")
            
            print(f"\n✅ Gradient Boosting plugin test completed successfully!")
            
        else:
            print(f"❌ Compatibility issue: {message}")
    
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()