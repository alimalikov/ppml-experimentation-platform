import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.ensemble import RandomForestRegressor
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


class RandomForestRegressorPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Random Forest Regressor Plugin - Robust Ensemble Method
    
    Random Forest is a powerful ensemble method that combines multiple decision trees
    to create a robust, accurate, and versatile regression model. It addresses the
    main weaknesses of individual decision trees (overfitting and instability) while
    maintaining their interpretability advantages.
    
    Key Features:
    - Robust ensemble of decision trees with bagging
    - Built-in cross-validation through out-of-bag (OOB) samples
    - Automatic feature importance ranking
    - Handles non-linear relationships without assumptions
    - Robust to outliers and noise
    - Parallel training for efficiency
    - Built-in regularization through randomness
    - Comprehensive ensemble analysis and tree diversity metrics
    - Advanced feature selection and interaction analysis
    - Extensive hyperparameter optimization guidance
    """
    
    def __init__(
        self,
        # Core ensemble parameters
        n_estimators=100,
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        
        # Randomness and diversity parameters
        max_features='sqrt',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True,
        
        # Control parameters
        n_jobs=-1,
        random_state=42,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        
        # Analysis options
        compute_feature_importance=True,
        compute_permutation_importance=True,
        compute_oob_analysis=True,
        tree_diversity_analysis=True,
        ensemble_convergence_analysis=True,
        
        # Advanced analysis
        feature_interaction_analysis=True,
        learning_curve_analysis=True,
        hyperparameter_sensitivity_analysis=True,
        ensemble_composition_analysis=True,
        
        # Visualization options
        visualize_trees=False,  # Only for small forests
        max_trees_to_visualize=3,
        feature_importance_analysis=True,
        prediction_distribution_analysis=True,
        
        # Performance analysis
        cross_validation_analysis=True,
        cv_folds=5,
        performance_benchmarking=True
    ):
        super().__init__()
        
        # Core ensemble parameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        
        # Randomness and diversity parameters
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        
        # Control parameters
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        
        # Analysis options
        self.compute_feature_importance = compute_feature_importance
        self.compute_permutation_importance = compute_permutation_importance
        self.compute_oob_analysis = compute_oob_analysis
        self.tree_diversity_analysis = tree_diversity_analysis
        self.ensemble_convergence_analysis = ensemble_convergence_analysis
        
        # Advanced analysis
        self.feature_interaction_analysis = feature_interaction_analysis
        self.learning_curve_analysis = learning_curve_analysis
        self.hyperparameter_sensitivity_analysis = hyperparameter_sensitivity_analysis
        self.ensemble_composition_analysis = ensemble_composition_analysis
        
        # Visualization options
        self.visualize_trees = visualize_trees
        self.max_trees_to_visualize = max_trees_to_visualize
        self.feature_importance_analysis = feature_importance_analysis
        self.prediction_distribution_analysis = prediction_distribution_analysis
        
        # Performance analysis
        self.cross_validation_analysis = cross_validation_analysis
        self.cv_folds = cv_folds
        self.performance_benchmarking = performance_benchmarking
        
        # Required plugin metadata
        self._name = "Random Forest Regressor"
        self._description = "Robust ensemble regression using multiple decision trees"
        self._category = "Ensemble Methods"
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 20
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        
        # Analysis results
        self.feature_importance_analysis_ = {}
        self.oob_analysis_ = {}
        self.tree_diversity_analysis_ = {}
        self.ensemble_convergence_analysis_ = {}
        self.feature_interaction_analysis_ = {}
        self.learning_curve_analysis_ = {}
        self.hyperparameter_sensitivity_analysis_ = {}
        self.ensemble_composition_analysis_ = {}
        self.cross_validation_analysis_ = {}
        self.performance_benchmarking_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Random Forest Regressor with comprehensive ensemble analysis
        
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
        
        # Create and configure Random Forest model
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples
        )
        
        # Fit the model
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        # Perform comprehensive ensemble analysis
        self._analyze_feature_importance()
        self._analyze_oob_performance()
        self._analyze_tree_diversity()
        self._analyze_ensemble_convergence()
        self._analyze_feature_interactions()
        self._analyze_learning_curves()
        self._analyze_hyperparameter_sensitivity()
        self._analyze_ensemble_composition()
        self._analyze_cross_validation()
        self._analyze_performance_benchmarks()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted Random Forest
        
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
    
    def predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty estimates using tree predictions
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        
        Returns:
        --------
        results : dict
            Dictionary containing predictions, uncertainty estimates, and tree predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.model_.estimators_])
        
        # Calculate statistics
        mean_predictions = np.mean(tree_predictions, axis=0)
        std_predictions = np.std(tree_predictions, axis=0)
        
        # Calculate prediction intervals (assuming normal distribution)
        confidence_95_lower = mean_predictions - 1.96 * std_predictions
        confidence_95_upper = mean_predictions + 1.96 * std_predictions
        
        # Calculate ensemble agreement (coefficient of variation)
        cv = std_predictions / (np.abs(mean_predictions) + 1e-10)
        
        return {
            'predictions': mean_predictions,
            'std_predictions': std_predictions,
            'confidence_95_lower': confidence_95_lower,
            'confidence_95_upper': confidence_95_upper,
            'prediction_interval_width': confidence_95_upper - confidence_95_lower,
            'coefficient_of_variation': cv,
            'tree_predictions': tree_predictions,
            'ensemble_agreement': 1.0 / (1.0 + cv),  # Higher is better agreement
            'uncertainty_score': cv  # Lower is more certain
        }
    
    def _analyze_feature_importance(self):
        """Analyze feature importance using multiple methods"""
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
                        scoring='neg_mean_squared_error', n_jobs=self.n_jobs
                    )
                    permutation_imp = perm_imp_result.importances_mean
                    permutation_imp_std = perm_imp_result.importances_std
                except:
                    permutation_imp = None
                    permutation_imp_std = None
            
            # Tree-wise feature importance analysis
            tree_importances = np.array([tree.feature_importances_ for tree in self.model_.estimators_])
            importance_stability = {
                'mean_importance': np.mean(tree_importances, axis=0),
                'std_importance': np.std(tree_importances, axis=0),
                'cv_importance': np.std(tree_importances, axis=0) / (np.mean(tree_importances, axis=0) + 1e-10),
                'consensus_importance': np.mean(tree_importances > 0, axis=0)  # Fraction of trees using each feature
            }
            
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
                'gini_coefficient': self._calculate_gini_coefficient(builtin_importance)
            }
            
            # Identify stable important features
            stable_threshold = np.mean(builtin_importance) + 0.5 * np.std(builtin_importance)
            consensus_threshold = 0.7  # Feature used in 70% of trees
            
            stable_important_features = (
                (builtin_importance > stable_threshold) & 
                (importance_stability['consensus_importance'] > consensus_threshold)
            )
            
            self.feature_importance_analysis_ = {
                'builtin_importance': builtin_importance,
                'permutation_importance': permutation_imp,
                'permutation_importance_std': permutation_imp_std,
                'importance_stability': importance_stability,
                'feature_ranking': importance_ranking,
                'feature_names': self.feature_names_,
                'importance_statistics': importance_stats,
                'stable_important_features': stable_important_features,
                'consensus_threshold': consensus_threshold,
                'stable_threshold': stable_threshold,
                'top_features': [
                    {
                        'name': self.feature_names_[i],
                        'builtin_importance': builtin_importance[i],
                        'permutation_importance': permutation_imp[i] if permutation_imp is not None else None,
                        'importance_std': importance_stability['std_importance'][i],
                        'consensus': importance_stability['consensus_importance'][i],
                        'stability_cv': importance_stability['cv_importance'][i]
                    }
                    for i in importance_ranking[:15]
                ]
            }
            
        except Exception as e:
            self.feature_importance_analysis_ = {
                'error': f'Could not analyze feature importance: {str(e)}'
            }
    
    def _calculate_importance_concentration(self, importance_values):
        """Calculate how concentrated the importance is in top features"""
        try:
            sorted_importance = np.sort(importance_values)[::-1]
            total_importance = np.sum(sorted_importance)
            
            # Calculate cumulative importance
            cumsum = np.cumsum(sorted_importance) / total_importance
            
            # Find features needed for 80% of importance
            features_80 = np.argmax(cumsum >= 0.8) + 1
            
            return {
                'top_5_concentration': np.sum(sorted_importance[:5]) / total_importance,
                'top_10_concentration': np.sum(sorted_importance[:10]) / total_importance,
                'features_for_80_percent': features_80,
                'features_for_90_percent': np.argmax(cumsum >= 0.9) + 1
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
    
    def _analyze_oob_performance(self):
        """Analyze out-of-bag (OOB) performance"""
        if not self.compute_oob_analysis or not self.oob_score:
            return
        
        try:
            # OOB score (R²)
            oob_r2 = self.model_.oob_score_
            
            # OOB predictions
            oob_predictions = self.model_.oob_prediction_
            
            # Calculate additional OOB metrics
            oob_mse = mean_squared_error(self.y_original_, oob_predictions)
            oob_mae = mean_absolute_error(self.y_original_, oob_predictions)
            oob_rmse = np.sqrt(oob_mse)
            
            # Training score for comparison
            train_predictions = self.model_.predict(self.X_original_)
            train_r2 = r2_score(self.y_original_, train_predictions)
            train_mse = mean_squared_error(self.y_original_, train_predictions)
            
            # Overfitting assessment
            overfitting_metrics = {
                'r2_gap': train_r2 - oob_r2,
                'mse_ratio': oob_mse / train_mse,
                'relative_r2_drop': (train_r2 - oob_r2) / train_r2 if train_r2 > 0 else float('inf')
            }
            
            # OOB performance assessment
            performance_assessment = self._assess_oob_performance(oob_r2, overfitting_metrics)
            
            self.oob_analysis_ = {
                'oob_score_r2': oob_r2,
                'oob_mse': oob_mse,
                'oob_mae': oob_mae,
                'oob_rmse': oob_rmse,
                'oob_predictions': oob_predictions,
                'training_performance': {
                    'train_r2': train_r2,
                    'train_mse': train_mse
                },
                'overfitting_metrics': overfitting_metrics,
                'performance_assessment': performance_assessment,
                'oob_advantage': "Built-in cross-validation without separate test set"
            }
            
        except Exception as e:
            self.oob_analysis_ = {
                'error': f'Could not analyze OOB performance: {str(e)}'
            }
    
    def _assess_oob_performance(self, oob_r2, overfitting_metrics):
        """Assess OOB performance and overfitting"""
        try:
            r2_gap = overfitting_metrics['r2_gap']
            
            # Performance categories
            if oob_r2 > 0.8 and r2_gap < 0.05:
                overall = "Excellent - High performance with minimal overfitting"
            elif oob_r2 > 0.6 and r2_gap < 0.1:
                overall = "Good - Solid performance with low overfitting"
            elif oob_r2 > 0.4 and r2_gap < 0.15:
                overall = "Fair - Moderate performance, acceptable overfitting"
            elif oob_r2 > 0.2:
                overall = "Poor - Low performance, check model complexity"
            else:
                overall = "Very Poor - Model not learning effectively"
            
            # Overfitting assessment
            if r2_gap < 0.05:
                overfitting = "Minimal overfitting"
            elif r2_gap < 0.1:
                overfitting = "Low overfitting"
            elif r2_gap < 0.2:
                overfitting = "Moderate overfitting"
            else:
                overfitting = "High overfitting - consider regularization"
            
            return {
                'overall_assessment': overall,
                'overfitting_assessment': overfitting,
                'oob_r2': oob_r2,
                'r2_gap': r2_gap,
                'generalization_score': max(0, 1 - r2_gap / 0.2)  # Score from 0-1
            }
            
        except:
            return {'overall_assessment': 'Unknown', 'overfitting_assessment': 'Unknown'}
    
    def _analyze_tree_diversity(self):
        """Analyze diversity among trees in the ensemble"""
        if not self.tree_diversity_analysis:
            return
        
        try:
            # Sample a subset of training data for diversity analysis
            sample_size = min(500, len(self.X_original_))
            indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
            X_sample = self.X_original_[indices]
            y_sample = self.y_original_[indices]
            
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X_sample) for tree in self.model_.estimators_])
            
            # Calculate pairwise correlations between tree predictions
            n_trees = len(self.model_.estimators_)
            correlations = []
            
            for i in range(min(n_trees, 50)):  # Limit for computational efficiency
                for j in range(i + 1, min(n_trees, 50)):
                    corr = np.corrcoef(tree_predictions[i], tree_predictions[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            correlations = np.array(correlations)
            
            # Calculate diversity metrics
            diversity_metrics = {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'min_correlation': np.min(correlations),
                'max_correlation': np.max(correlations),
                'diversity_score': 1 - np.mean(correlations),  # Higher is more diverse
                'low_correlation_pairs': np.sum(correlations < 0.5) / len(correlations)
            }
            
            # Analyze tree structure diversity
            tree_depths = [tree.tree_.max_depth for tree in self.model_.estimators_]
            tree_nodes = [tree.tree_.node_count for tree in self.model_.estimators_]
            tree_leaves = [tree.tree_.n_leaves for tree in self.model_.estimators_]
            
            structure_diversity = {
                'depth_diversity': {
                    'mean': np.mean(tree_depths),
                    'std': np.std(tree_depths),
                    'range': np.max(tree_depths) - np.min(tree_depths)
                },
                'node_diversity': {
                    'mean': np.mean(tree_nodes),
                    'std': np.std(tree_nodes),
                    'cv': np.std(tree_nodes) / np.mean(tree_nodes)
                },
                'leaf_diversity': {
                    'mean': np.mean(tree_leaves),
                    'std': np.std(tree_leaves),
                    'cv': np.std(tree_leaves) / np.mean(tree_leaves)
                }
            }
            
            # Diversity assessment
            diversity_assessment = self._assess_ensemble_diversity(diversity_metrics, structure_diversity)
            
            self.tree_diversity_analysis_ = {
                'prediction_diversity': diversity_metrics,
                'structure_diversity': structure_diversity,
                'diversity_assessment': diversity_assessment,
                'n_trees_analyzed': min(n_trees, 50),
                'n_samples_used': sample_size
            }
            
        except Exception as e:
            self.tree_diversity_analysis_ = {
                'error': f'Could not analyze tree diversity: {str(e)}'
            }
    
    def _assess_ensemble_diversity(self, diversity_metrics, structure_diversity):
        """Assess the diversity of the ensemble"""
        try:
            diversity_score = diversity_metrics['diversity_score']
            mean_correlation = diversity_metrics['mean_correlation']
            
            # Diversity categories
            if diversity_score > 0.7:
                diversity_level = "Very High - Excellent ensemble diversity"
            elif diversity_score > 0.5:
                diversity_level = "High - Good ensemble diversity"
            elif diversity_score > 0.3:
                diversity_level = "Moderate - Acceptable diversity"
            elif diversity_score > 0.1:
                diversity_level = "Low - Limited diversity, may underperform"
            else:
                diversity_level = "Very Low - Poor diversity, consider increasing randomness"
            
            # Recommendations
            recommendations = []
            if diversity_score < 0.3:
                recommendations.extend([
                    "Increase max_features randomness",
                    "Consider reducing max_depth",
                    "Increase bootstrap sample diversity",
                    "Add more trees to ensemble"
                ])
            elif diversity_score < 0.5:
                recommendations.append("Consider slight increase in randomness parameters")
            else:
                recommendations.append("Diversity level is good for ensemble performance")
            
            return {
                'diversity_level': diversity_level,
                'diversity_score': diversity_score,
                'mean_correlation': mean_correlation,
                'recommendations': recommendations,
                'optimal_range': 'Diversity score between 0.4-0.7 is typically optimal'
            }
            
        except:
            return {'diversity_level': 'Unknown', 'recommendations': []}
    
    def _analyze_ensemble_convergence(self):
        """Analyze how ensemble performance converges with number of trees"""
        if not self.ensemble_convergence_analysis:
            return
        
        try:
            # Test different numbers of trees
            tree_counts = [1, 5, 10, 25, 50, 75] + list(range(100, self.n_estimators + 1, max(25, self.n_estimators // 10)))
            tree_counts = [t for t in tree_counts if t <= self.n_estimators]
            
            convergence_scores = []
            oob_scores = []
            
            for n_trees in tree_counts:
                try:
                    # Create limited ensemble
                    limited_estimators = self.model_.estimators_[:n_trees]
                    
                    # Calculate ensemble prediction
                    tree_preds = np.array([tree.predict(self.X_original_) for tree in limited_estimators])
                    ensemble_pred = np.mean(tree_preds, axis=0)
                    
                    # Calculate performance
                    r2 = r2_score(self.y_original_, ensemble_pred)
                    convergence_scores.append(r2)
                    
                    # OOB score approximation (if available)
                    if self.oob_score and hasattr(self.model_, 'oob_prediction_'):
                        # Approximate OOB score with subset
                        oob_r2 = r2_score(self.y_original_, ensemble_pred)  # Simplified
                        oob_scores.append(oob_r2)
                    
                except:
                    convergence_scores.append(np.nan)
                    if oob_scores:
                        oob_scores.append(np.nan)
            
            # Analyze convergence patterns
            convergence_analysis = self._analyze_convergence_pattern(tree_counts, convergence_scores)
            
            self.ensemble_convergence_analysis_ = {
                'tree_counts': tree_counts,
                'training_scores': convergence_scores,
                'oob_scores': oob_scores if oob_scores else None,
                'convergence_analysis': convergence_analysis,
                'optimal_n_estimators': convergence_analysis.get('optimal_trees', self.n_estimators)
            }
            
        except Exception as e:
            self.ensemble_convergence_analysis_ = {
                'error': f'Could not analyze ensemble convergence: {str(e)}'
            }
    
    def _analyze_convergence_pattern(self, tree_counts, scores):
        """Analyze the convergence pattern of ensemble performance"""
        try:
            valid_scores = [(t, s) for t, s in zip(tree_counts, scores) if not np.isnan(s)]
            
            if len(valid_scores) < 3:
                return {'status': 'Insufficient data for convergence analysis'}
            
            tree_counts_valid, scores_valid = zip(*valid_scores)
            scores_valid = np.array(scores_valid)
            
            # Find point of diminishing returns
            score_improvements = np.diff(scores_valid)
            
            # Find where improvement becomes small
            threshold = 0.001  # 0.1% improvement
            diminishing_point = None
            
            for i, improvement in enumerate(score_improvements):
                if improvement < threshold:
                    diminishing_point = tree_counts_valid[i + 1]
                    break
            
            # Calculate convergence metrics
            final_score = scores_valid[-1]
            max_score = np.max(scores_valid)
            convergence_ratio = final_score / max_score if max_score > 0 else 0
            
            # Stability of final scores
            last_5_scores = scores_valid[-5:] if len(scores_valid) >= 5 else scores_valid
            score_stability = np.std(last_5_scores)
            
            return {
                'converged': score_stability < 0.01,
                'diminishing_returns_at': diminishing_point,
                'final_score': final_score,
                'max_score': max_score,
                'convergence_ratio': convergence_ratio,
                'score_stability': score_stability,
                'optimal_trees': diminishing_point if diminishing_point else self.n_estimators,
                'recommendation': self._get_convergence_recommendation(diminishing_point, self.n_estimators)
            }
            
        except:
            return {'status': 'Error in convergence analysis'}
    
    def _get_convergence_recommendation(self, diminishing_point, current_trees):
        """Get recommendation based on convergence analysis"""
        try:
            if diminishing_point is None:
                return f"Consider increasing n_estimators beyond {current_trees} for potential improvement"
            elif diminishing_point < current_trees * 0.5:
                return f"Performance plateaus early at ~{diminishing_point} trees. Consider reducing n_estimators for efficiency"
            elif diminishing_point < current_trees * 0.8:
                return f"Good balance achieved around {diminishing_point} trees. Current setting is reasonable"
            else:
                return f"Performance still improving. Current setting of {current_trees} trees is appropriate"
        except:
            return "Could not generate convergence recommendation"
    
    def _analyze_feature_interactions(self):
        """Analyze feature interactions in the ensemble"""
        if not self.feature_interaction_analysis:
            return
        
        try:
            # Sample subset for computational efficiency
            sample_size = min(200, len(self.X_original_))
            indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
            X_sample = self.X_original_[indices]
            
            # Analyze two-way interactions using partial dependence
            top_features = np.argsort(self.model_.feature_importances_)[-5:]  # Top 5 features
            
            interactions = []
            
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    try:
                        # Simple interaction analysis: predict with one feature perturbed
                        interaction_strength = self._calculate_interaction_strength(
                            X_sample, feat1, feat2
                        )
                        
                        interactions.append({
                            'feature1': self.feature_names_[feat1],
                            'feature2': self.feature_names_[feat2],
                            'feature1_idx': feat1,
                            'feature2_idx': feat2,
                            'interaction_strength': interaction_strength,
                            'feature1_importance': self.model_.feature_importances_[feat1],
                            'feature2_importance': self.model_.feature_importances_[feat2]
                        })
                    except:
                        continue
            
            # Sort by interaction strength
            interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            self.feature_interaction_analysis_ = {
                'interactions': interactions[:10],  # Top 10 interactions
                'n_features_analyzed': len(top_features),
                'sample_size_used': sample_size,
                'top_interactions': [
                    f"{inter['feature1']} × {inter['feature2']} (strength: {inter['interaction_strength']:.3f})"
                    for inter in interactions[:5]
                ]
            }
            
        except Exception as e:
            self.feature_interaction_analysis_ = {
                'error': f'Could not analyze feature interactions: {str(e)}'
            }
    
    def _calculate_interaction_strength(self, X_sample, feat1_idx, feat2_idx):
        """Calculate interaction strength between two features"""
        try:
            # Baseline predictions
            baseline_pred = self.model_.predict(X_sample)
            
            # Perturb feature 1
            X_perturb1 = X_sample.copy()
            X_perturb1[:, feat1_idx] = np.median(X_sample[:, feat1_idx])
            pred1 = self.model_.predict(X_perturb1)
            
            # Perturb feature 2
            X_perturb2 = X_sample.copy()
            X_perturb2[:, feat2_idx] = np.median(X_sample[:, feat2_idx])
            pred2 = self.model_.predict(X_perturb2)
            
            # Perturb both features
            X_perturb_both = X_sample.copy()
            X_perturb_both[:, feat1_idx] = np.median(X_sample[:, feat1_idx])
            X_perturb_both[:, feat2_idx] = np.median(X_sample[:, feat2_idx])
            pred_both = self.model_.predict(X_perturb_both)
            
            # Calculate interaction as deviation from additive effect
            additive_effect = (pred1 - baseline_pred) + (pred2 - baseline_pred)
            combined_effect = pred_both - baseline_pred
            interaction = np.mean(np.abs(combined_effect - additive_effect))
            
            return interaction
            
        except:
            return 0.0
    
    def _analyze_learning_curves(self):
        """Analyze learning curves for different training set sizes"""
        if not self.learning_curve_analysis:
            return
        
        try:
            # Define training sizes
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            # Create simplified model for learning curve analysis
            simple_model = RandomForestRegressor(
                n_estimators=min(50, self.n_estimators),
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=1  # Avoid nested parallelism
            )
            
            # Calculate learning curves
            train_sizes_abs, train_scores, val_scores = learning_curve(
                simple_model, self.X_original_, self.y_original_,
                train_sizes=train_sizes,
                cv=3,  # Reduced for efficiency
                scoring='r2',
                n_jobs=1,
                random_state=self.random_state
            )
            
            # Calculate statistics
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            val_scores_mean = np.mean(val_scores, axis=1)
            val_scores_std = np.std(val_scores, axis=1)
            
            # Analyze learning curve characteristics
            learning_analysis = self._analyze_learning_characteristics(
                train_sizes_abs, train_scores_mean, val_scores_mean
            )
            
            self.learning_curve_analysis_ = {
                'train_sizes': train_sizes_abs,
                'train_scores_mean': train_scores_mean,
                'train_scores_std': train_scores_std,
                'val_scores_mean': val_scores_mean,
                'val_scores_std': val_scores_std,
                'learning_analysis': learning_analysis
            }
            
        except Exception as e:
            self.learning_curve_analysis_ = {
                'error': f'Could not analyze learning curves: {str(e)}'
            }
    
    def _analyze_learning_characteristics(self, train_sizes, train_scores, val_scores):
        """Analyze characteristics of the learning curves"""
        try:
            # Final performance gap
            final_gap = train_scores[-1] - val_scores[-1]
            
            # Convergence analysis
            val_improvements = np.diff(val_scores)
            converged = np.all(val_improvements[-3:] < 0.01) if len(val_improvements) >= 3 else False
            
            # Data efficiency
            half_data_performance = val_scores[len(val_scores)//2] if len(val_scores) > 2 else 0
            full_data_performance = val_scores[-1]
            data_efficiency = half_data_performance / full_data_performance if full_data_performance > 0 else 0
            
            return {
                'final_gap': final_gap,
                'converged': converged,
                'data_efficiency': data_efficiency,
                'best_val_score': np.max(val_scores),
                'optimal_training_size': train_sizes[np.argmax(val_scores)],
                'overfitting_level': 'High' if final_gap > 0.2 else 'Medium' if final_gap > 0.1 else 'Low'
            }
            
        except:
            return {'status': 'Error in learning curve analysis'}
    
    def _analyze_hyperparameter_sensitivity(self):
        """Analyze sensitivity to key hyperparameters"""
        if not self.hyperparameter_sensitivity_analysis:
            return
        
        try:
            sensitivity_results = {}
            
            # Analyze n_estimators sensitivity
            if self.n_estimators >= 50:
                n_est_values = [25, 50, 100, 200] if self.n_estimators >= 200 else [25, 50, self.n_estimators]
                n_est_sensitivity = self._test_hyperparameter_values(
                    'n_estimators', n_est_values, max_tests=3
                )
                sensitivity_results['n_estimators'] = n_est_sensitivity
            
            # Analyze max_depth sensitivity
            depth_values = [3, 5, 10, None] if self.max_depth is None else [3, self.max_depth, None]
            depth_sensitivity = self._test_hyperparameter_values(
                'max_depth', depth_values, max_tests=3
            )
            sensitivity_results['max_depth'] = depth_sensitivity
            
            # Analyze max_features sensitivity
            feature_values = ['sqrt', 'log2', None] if self.max_features != 'sqrt' else ['sqrt', None]
            feature_sensitivity = self._test_hyperparameter_values(
                'max_features', feature_values, max_tests=2
            )
            sensitivity_results['max_features'] = feature_sensitivity
            
            # Overall sensitivity assessment
            sensitivity_assessment = self._assess_hyperparameter_sensitivity(sensitivity_results)
            
            self.hyperparameter_sensitivity_analysis_ = {
                'sensitivity_results': sensitivity_results,
                'sensitivity_assessment': sensitivity_assessment
            }
            
        except Exception as e:
            self.hyperparameter_sensitivity_analysis_ = {
                'error': f'Could not analyze hyperparameter sensitivity: {str(e)}'
            }
    
    def _test_hyperparameter_values(self, param_name, param_values, max_tests=3):
        """Test different values for a hyperparameter"""
        try:
            results = []
            
            for value in param_values[:max_tests]:
                try:
                    # Create model with modified parameter
                    params = {
                        'n_estimators': min(30, self.n_estimators),  # Reduced for speed
                        'random_state': self.random_state,
                        'n_jobs': 1
                    }
                    params[param_name] = value
                    
                    model = RandomForestRegressor(**params)
                    
                    # Cross-validation score
                    scores = cross_val_score(
                        model, self.X_original_, self.y_original_,
                        cv=3, scoring='r2'
                    )
                    
                    results.append({
                        'value': value,
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores)
                    })
                    
                except:
                    continue
            
            return {
                'results': results,
                'best_value': max(results, key=lambda x: x['mean_score'])['value'] if results else None,
                'score_range': max(r['mean_score'] for r in results) - min(r['mean_score'] for r in results) if results else 0
            }
            
        except:
            return {'results': [], 'error': 'Could not test parameter values'}
    
    def _assess_hyperparameter_sensitivity(self, sensitivity_results):
        """Assess overall hyperparameter sensitivity"""
        try:
            sensitive_params = []
            stable_params = []
            
            for param, result in sensitivity_results.items():
                if 'score_range' in result:
                    if result['score_range'] > 0.1:
                        sensitive_params.append(param)
                    else:
                        stable_params.append(param)
            
            return {
                'sensitive_parameters': sensitive_params,
                'stable_parameters': stable_params,
                'overall_sensitivity': 'High' if len(sensitive_params) > 2 else 'Medium' if sensitive_params else 'Low',
                'tuning_priority': sensitive_params[:2] if sensitive_params else ['n_estimators', 'max_depth']
            }
            
        except:
            return {'overall_sensitivity': 'Unknown'}
    
    def _analyze_ensemble_composition(self):
        """Analyze the composition and characteristics of the ensemble"""
        if not self.ensemble_composition_analysis:
            return
        
        try:
            # Tree characteristics
            tree_depths = [tree.tree_.max_depth for tree in self.model_.estimators_]
            tree_nodes = [tree.tree_.node_count for tree in self.model_.estimators_]
            tree_leaves = [tree.tree_.n_leaves for tree in self.model_.estimators_]
            
            # Feature usage across trees
            feature_usage = np.zeros(self.n_features_in_)
            for tree in self.model_.estimators_:
                used_features = np.unique(tree.tree_.feature[tree.tree_.feature >= 0])
                feature_usage[used_features] += 1
            
            feature_usage_freq = feature_usage / len(self.model_.estimators_)
            
            # Bootstrap sample analysis (if bootstrap is True)
            bootstrap_diversity = None
            if self.bootstrap:
                # Estimate bootstrap diversity
                n_samples = len(self.y_original_)
                expected_unique = n_samples * (1 - (1 - 1/n_samples)**n_samples)
                bootstrap_diversity = {
                    'expected_unique_samples_per_tree': expected_unique,
                    'sample_overlap_estimate': 1 - expected_unique / n_samples
                }
            
            composition_summary = {
                'tree_statistics': {
                    'mean_depth': np.mean(tree_depths),
                    'std_depth': np.std(tree_depths),
                    'mean_nodes': np.mean(tree_nodes),
                    'std_nodes': np.std(tree_nodes),
                    'mean_leaves': np.mean(tree_leaves),
                    'std_leaves': np.std(tree_leaves)
                },
                'feature_usage': {
                    'mean_usage_frequency': np.mean(feature_usage_freq),
                    'std_usage_frequency': np.std(feature_usage_freq),
                    'most_used_features': np.argsort(feature_usage_freq)[-5:],
                    'least_used_features': np.argsort(feature_usage_freq)[:5],
                    'unused_features': np.sum(feature_usage_freq == 0)
                },
                'bootstrap_diversity': bootstrap_diversity,
                'ensemble_size': len(self.model_.estimators_),
                'total_nodes': np.sum(tree_nodes),
                'total_leaves': np.sum(tree_leaves)
            }
            
            self.ensemble_composition_analysis_ = composition_summary
            
        except Exception as e:
            self.ensemble_composition_analysis_ = {
                'error': f'Could not analyze ensemble composition: {str(e)}'
            }
    
    def _analyze_cross_validation(self):
        """Perform comprehensive cross-validation analysis"""
        if not self.cross_validation_analysis:
            return
        
        try:
            # Cross-validation with multiple metrics
            scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            cv_results = {}
            
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(
                        self.model_, self.X_original_, self.y_original_,
                        cv=self.cv_folds, scoring=metric, n_jobs=self.n_jobs
                    )
                    
                    cv_results[metric] = {
                        'scores': scores,
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
                except:
                    continue
            
            # Training performance for comparison
            train_pred = self.model_.predict(self.X_original_)
            train_r2 = r2_score(self.y_original_, train_pred)
            train_mse = mean_squared_error(self.y_original_, train_pred)
            train_mae = mean_absolute_error(self.y_original_, train_pred)
            
            # Generalization analysis
            if 'r2' in cv_results:
                generalization_gap = train_r2 - cv_results['r2']['mean']
                
                performance_assessment = {
                    'cv_r2_mean': cv_results['r2']['mean'],
                    'train_r2': train_r2,
                    'generalization_gap': generalization_gap,
                    'cv_stability': cv_results['r2']['std'],
                    'performance_consistency': 'High' if cv_results['r2']['std'] < 0.05 else 'Medium' if cv_results['r2']['std'] < 0.1 else 'Low'
                }
            else:
                performance_assessment = {'error': 'Could not calculate R² scores'}
            
            self.cross_validation_analysis_ = {
                'cv_results': cv_results,
                'training_performance': {
                    'train_r2': train_r2,
                    'train_mse': train_mse,
                    'train_mae': train_mae
                },
                'performance_assessment': performance_assessment,
                'cv_folds': self.cv_folds
            }
            
        except Exception as e:
            self.cross_validation_analysis_ = {
                'error': f'Could not perform cross-validation analysis: {str(e)}'
            }
    
    def _analyze_performance_benchmarks(self):
        """Analyze performance against simple benchmarks"""
        if not self.performance_benchmarking:
            return
        
        try:
            from sklearn.dummy import DummyRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.tree import DecisionTreeRegressor
            
            # Create benchmark models
            benchmarks = {
                'mean_predictor': DummyRegressor(strategy='mean'),
                'median_predictor': DummyRegressor(strategy='median'),
                'linear_regression': LinearRegression(),
                'single_tree': DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            }
            
            benchmark_results = {}
            
            for name, model in benchmarks.items():
                try:
                    scores = cross_val_score(
                        model, self.X_original_, self.y_original_,
                        cv=min(3, self.cv_folds), scoring='r2'
                    )
                    
                    benchmark_results[name] = {
                        'cv_r2_mean': np.mean(scores),
                        'cv_r2_std': np.std(scores)
                    }
                except:
                    benchmark_results[name] = {'error': 'Failed to evaluate'}
            
            # Random Forest performance
            if 'r2' in self.cross_validation_analysis_.get('cv_results', {}):
                rf_performance = self.cross_validation_analysis_['cv_results']['r2']['mean']
            else:
                rf_performance = self.model_.score(self.X_original_, self.y_original_)
            
            # Performance improvements
            improvements = {}
            for name, result in benchmark_results.items():
                if 'cv_r2_mean' in result:
                    improvement = rf_performance - result['cv_r2_mean']
                    improvements[name] = {
                        'absolute_improvement': improvement,
                        'relative_improvement': improvement / max(0.01, abs(result['cv_r2_mean']))
                    }
            
            self.performance_benchmarking_ = {
                'random_forest_r2': rf_performance,
                'benchmark_results': benchmark_results,
                'improvements': improvements,
                'performance_ranking': self._rank_performance(rf_performance, benchmark_results)
            }
            
        except Exception as e:
            self.performance_benchmarking_ = {
                'error': f'Could not analyze performance benchmarks: {str(e)}'
            }
    
    def _rank_performance(self, rf_score, benchmark_results):
        """Rank Random Forest performance against benchmarks"""
        try:
            all_scores = [rf_score]
            model_names = ['Random Forest']
            
            for name, result in benchmark_results.items():
                if 'cv_r2_mean' in result:
                    all_scores.append(result['cv_r2_mean'])
                    model_names.append(name.replace('_', ' ').title())
            
            # Sort by score
            sorted_indices = np.argsort(all_scores)[::-1]
            ranking = [(model_names[i], all_scores[i]) for i in sorted_indices]
            
            rf_rank = next(i for i, (name, _) in enumerate(ranking) if name == 'Random Forest') + 1
            
            return {
                'ranking': ranking,
                'random_forest_rank': rf_rank,
                'total_models': len(ranking),
                'performance_percentile': (len(ranking) - rf_rank + 1) / len(ranking) * 100
            }
            
        except:
            return {'error': 'Could not rank performance'}
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Ensemble Configuration", "Tree Parameters", "Randomness & Diversity", "Analysis Options", "Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Random Forest Ensemble Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.number_input(
                    "Number of Trees:",
                    value=self.n_estimators,
                    min_value=10,
                    max_value=500,
                    step=10,
                    help="Number of trees in the forest",
                    key=f"{key_prefix}_n_estimators"
                )
                
                criterion = st.selectbox(
                    "Split Criterion:",
                    options=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    index=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'].index(self.criterion),
                    help="Function to measure split quality",
                    key=f"{key_prefix}_criterion"
                )
                
                bootstrap = st.checkbox(
                    "Bootstrap Sampling",
                    value=self.bootstrap,
                    help="Whether bootstrap samples are used when building trees",
                    key=f"{key_prefix}_bootstrap"
                )
                
                oob_score = st.checkbox(
                    "Out-of-Bag Score",
                    value=self.oob_score if self.bootstrap else False,
                    help="Whether to use out-of-bag samples to estimate R²",
                    disabled=not bootstrap,
                    key=f"{key_prefix}_oob_score"
                )
            
            with col2:
                max_samples = st.selectbox(
                    "Max Samples per Tree:",
                    options=[None, 0.7, 0.8, 0.9, 1.0],
                    index=0 if self.max_samples is None else [None, 0.7, 0.8, 0.9, 1.0].index(self.max_samples),
                    help="Maximum number of samples to draw for each tree",
                    key=f"{key_prefix}_max_samples"
                )
                
                n_jobs = st.selectbox(
                    "Parallel Jobs:",
                    options=[-1, 1, 2, 4],
                    index=[-1, 1, 2, 4].index(self.n_jobs) if self.n_jobs in [-1, 1, 2, 4] else 0,
                    help="Number of jobs for parallel processing (-1 = all cores)",
                    key=f"{key_prefix}_n_jobs"
                )
                
                random_state = st.number_input(
                    "Random Seed:",
                    value=int(self.random_state),
                    min_value=0,
                    max_value=1000,
                    help="For reproducible results",
                    key=f"{key_prefix}_random_state"
                )
                
                warm_start = st.checkbox(
                    "Warm Start",
                    value=self.warm_start,
                    help="Reuse previous solution to add more estimators",
                    key=f"{key_prefix}_warm_start"
                )
        
        with tab2:
            st.markdown("**Individual Tree Parameters**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_depth = st.selectbox(
                    "Maximum Tree Depth:",
                    options=[None, 3, 5, 7, 10, 15, 20],
                    index=0 if self.max_depth is None else [None, 3, 5, 7, 10, 15, 20].index(self.max_depth),
                    help="Maximum depth of individual trees (None = unlimited)",
                    key=f"{key_prefix}_max_depth"
                )
                
                min_samples_split = st.number_input(
                    "Min Samples Split:",
                    value=self.min_samples_split,
                    min_value=2,
                    max_value=50,
                    step=1,
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
            
            with col2:
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
            st.markdown("**Randomness and Diversity Control**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_features = st.selectbox(
                    "Max Features per Split:",
                    options=['sqrt', 'log2', None, 0.3, 0.5, 0.7],
                    index=['sqrt', 'log2', None, 0.3, 0.5, 0.7].index(self.max_features) if self.max_features in ['sqrt', 'log2', None, 0.3, 0.5, 0.7] else 0,
                    help="Number of features to consider for best split",
                    key=f"{key_prefix}_max_features"
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
                st.markdown("**Diversity Guide:**")
                st.info("""
                **Increasing Diversity:**
                • **max_features='sqrt'**: Good default for most cases
                • **max_features='log2'**: Higher diversity, may reduce accuracy
                • **max_features=None**: All features, lower diversity
                • **Lower max_depth**: Forces different tree structures
                • **Bootstrap=True**: Essential for ensemble diversity
                • **Higher min_samples_split**: Creates more varied trees
                """)
        
        with tab4:
            st.markdown("**Analysis and Performance Options**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                compute_feature_importance = st.checkbox(
                    "Feature Importance Analysis",
                    value=self.compute_feature_importance,
                    help="Compute ensemble feature importance",
                    key=f"{key_prefix}_compute_feature_importance"
                )
                
                compute_permutation_importance = st.checkbox(
                    "Permutation Importance",
                    value=self.compute_permutation_importance,
                    help="Compute permutation-based feature importance",
                    key=f"{key_prefix}_compute_permutation_importance"
                )
                
                compute_oob_analysis = st.checkbox(
                    "Out-of-Bag Analysis",
                    value=self.compute_oob_analysis and oob_score,
                    help="Analyze out-of-bag performance",
                    disabled=not oob_score,
                    key=f"{key_prefix}_compute_oob_analysis"
                )
                
                                
                tree_diversity_analysis = st.checkbox(
                    "Tree Diversity Analysis",
                    value=self.tree_diversity_analysis,
                    help="Analyze diversity among ensemble trees",
                    key=f"{key_prefix}_tree_diversity_analysis"
                )
                
                ensemble_convergence_analysis = st.checkbox(
                    "Ensemble Convergence Analysis",
                    value=self.ensemble_convergence_analysis,
                    help="Analyze how performance converges with tree count",
                    key=f"{key_prefix}_ensemble_convergence_analysis"
                )
                
                feature_interaction_analysis = st.checkbox(
                    "Feature Interaction Analysis",
                    value=self.feature_interaction_analysis,
                    help="Analyze feature interactions in ensemble",
                    key=f"{key_prefix}_feature_interaction_analysis"
                )
            
            with col2:
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
                
                ensemble_composition_analysis = st.checkbox(
                    "Ensemble Composition Analysis",
                    value=self.ensemble_composition_analysis,
                    help="Analyze tree characteristics and composition",
                    key=f"{key_prefix}_ensemble_composition_analysis"
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
                help="Visualize structure of individual trees (for small forests)",
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
            st.markdown("**Random Forest Regressor - Ensemble Method**")
            
            # Algorithm information
            if st.button("📚 Algorithm Information", key=f"{key_prefix}_algo_info"):
                st.markdown("""
                **Random Forest Regression - Robust Ensemble Method**
                
                Random Forest is a powerful ensemble learning method that combines multiple decision trees 
                to create robust, accurate predictions while addressing the main weaknesses of individual trees.
                
                **Core Principles:**
                • **Bootstrap Aggregating (Bagging)** - Each tree trained on random bootstrap sample
                • **Feature Randomness** - Random subset of features considered at each split
                • **Ensemble Averaging** - Final prediction is average of all tree predictions
                • **Out-of-Bag Evaluation** - Built-in cross-validation using unused samples
                • **Variance Reduction** - Multiple trees reduce prediction variance
                
                **Key Advantages:**
                • 🎯 **Robust Performance** - Less prone to overfitting than single trees
                • 📊 **Feature Importance** - Automatic ranking of feature relevance
                • 🔍 **Uncertainty Estimation** - Tree disagreement indicates prediction confidence
                • ⚡ **Parallel Training** - Trees can be trained independently
                • 🛡️ **Robust to Outliers** - Ensemble effect reduces outlier impact
                • 📈 **Handles Non-linearity** - Captures complex relationships naturally
                """)
            
            # When to use Random Forest
            if st.button("🎯 When to Use Random Forest", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • Need robust, reliable predictions with uncertainty estimates
                • Non-linear relationships suspected or confirmed
                • Feature importance insights required
                • Ensemble method preferred over single model
                • Moderate interpretability acceptable (not black box, not fully transparent)
                
                **Data Characteristics:**
                • Medium to large datasets (100s to 100,000s of samples)
                • Mixed feature types (numerical and categorical)
                • Presence of irrelevant features (automatic selection)
                • Potential outliers or noise in data
                • Feature interactions suspected
                
                **Examples:**
                • Real estate price prediction with diverse features
                • Financial risk modeling with multiple indicators
                • Bioinformatics with gene expression data
                • Customer lifetime value prediction
                • Environmental modeling (pollution, weather)
                • Manufacturing quality prediction
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Excellent generalization performance (reduces overfitting)
                ✅ Built-in feature importance ranking
                ✅ Robust to outliers and missing values
                ✅ Handles mixed data types without preprocessing
                ✅ Provides uncertainty estimates through tree disagreement
                ✅ Built-in cross-validation (OOB samples)
                ✅ Parallel training for computational efficiency
                ✅ Automatic feature selection through randomness
                ✅ Less sensitive to hyperparameters than individual trees
                ✅ Can handle large numbers of features
                
                **Limitations:**
                ❌ Less interpretable than single decision trees
                ❌ Can overfit with very noisy data
                ❌ Biased toward categorical features with many categories
                ❌ Memory intensive (stores multiple trees)
                ❌ Prediction time slower than single models
                ❌ May not perform well on very small datasets
                ❌ Can struggle with extrapolation beyond training range
                """)
            
            # Random Forest vs other methods
            if st.button("🔍 Random Forest vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **Random Forest vs Other Regression Methods:**
                
                **Random Forest vs Single Decision Tree:**
                • RF: Much better generalization, less overfitting
                • Tree: More interpretable, faster training/prediction
                • RF: Better performance on most datasets
                • Tree: Better for simple, interpretable rules
                
                **Random Forest vs Linear Regression:**
                • RF: Handles non-linearity and interactions automatically
                • Linear: More interpretable, better extrapolation
                • RF: No assumptions about relationships
                • Linear: Better for understanding individual feature effects
                
                **Random Forest vs Gradient Boosting:**
                • RF: More robust, less prone to overfitting
                • Boosting: Often higher accuracy, more sensitive to hyperparameters
                • RF: Easier to tune, parallel training
                • Boosting: Better for competitions, requires more careful tuning
                
                **Random Forest vs Neural Networks:**
                • RF: Better interpretability, works well with tabular data
                • NN: Better for complex patterns, images, sequences
                • RF: Requires less data, easier to train
                • NN: More flexible, can model any function
                
                **Random Forest vs SVM:**
                • RF: Better scalability, automatic feature selection
                • SVM: Better theoretical foundation, kernel flexibility
                • RF: Handles categorical features naturally
                • SVM: Better for high-dimensional sparse data
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Random Forest Best Practices:**
                
                **Ensemble Configuration:**
                1. **n_estimators**: Start with 100, increase to 200-500 for better performance
                2. **max_features**: Use 'sqrt' (default) for most cases, 'log2' for more diversity
                3. **bootstrap=True**: Essential for ensemble diversity
                4. **oob_score=True**: Use for built-in validation
                
                **Tree Parameters:**
                1. **max_depth**: Usually leave as None, use min_samples_split for control
                2. **min_samples_split**: 2-20 depending on dataset size
                3. **min_samples_leaf**: 1-5, higher values reduce overfitting
                4. **ccp_alpha**: Use for post-pruning if overfitting detected
                
                **Performance Optimization:**
                1. Use **n_jobs=-1** for parallel training
                2. Monitor OOB score vs training score for overfitting
                3. Use learning curves to determine optimal training set size
                4. Analyze feature importance for feature selection
                
                **Model Validation:**
                1. Use OOB score for initial assessment
                2. Perform cross-validation for robust evaluation
                3. Analyze tree diversity for ensemble quality
                4. Check convergence to determine optimal n_estimators
                
                **Interpretability:**
                1. Use feature importance plots for insights
                2. Analyze partial dependence for feature effects
                3. Use SHAP values for individual predictions
                4. Examine tree diversity for ensemble behavior
                """)
            
            # Advanced techniques
            if st.button("🚀 Advanced Techniques", key=f"{key_prefix}_advanced"):
                st.markdown("""
                **Advanced Random Forest Techniques:**
                
                **Ensemble Optimization:**
                • **Dynamic Ensemble Sizing**: Use convergence analysis to find optimal n_estimators
                • **Feature Subset Optimization**: Tune max_features based on dataset characteristics
                • **Weighted Voting**: Weight trees based on OOB performance
                • **Ensemble Pruning**: Remove poor-performing trees post-training
                
                **Feature Engineering:**
                • **Feature Interaction Detection**: Use Random Forest to identify interactions
                • **Automated Feature Selection**: Use importance scores for feature pruning
                • **Feature Binning**: Discretize continuous features for interpretability
                • **Synthetic Features**: Create interaction terms for important feature pairs
                
                **Uncertainty Quantification:**
                • **Prediction Intervals**: Use tree disagreement for confidence bounds
                • **Quantile Regression**: Modify trees to predict quantiles
                • **Monte Carlo Dropout**: Add randomness during prediction
                • **Ensemble Diversity Metrics**: Monitor prediction variance
                
                **Specialized Variants:**
                • **Extremely Randomized Trees**: More randomness in splitting
                • **Balanced Random Forest**: Handle imbalanced datasets
                • **Isolation Forest**: Use for anomaly detection
                • **Multi-output Random Forest**: Handle multiple targets simultaneously
                
                **Hyperparameter Tuning:**
                • **Grid Search**: Systematic parameter exploration
                • **Random Search**: Efficient for high-dimensional spaces
                • **Bayesian Optimization**: Sample-efficient tuning
                • **Multi-objective Optimization**: Balance accuracy vs interpretability
                """)
        
        return {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": 0,
            "warm_start": warm_start,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "compute_feature_importance": compute_feature_importance,
            "compute_permutation_importance": compute_permutation_importance,
            "compute_oob_analysis": compute_oob_analysis,
            "tree_diversity_analysis": tree_diversity_analysis,
            "ensemble_convergence_analysis": ensemble_convergence_analysis,
            "feature_interaction_analysis": feature_interaction_analysis,
            "learning_curve_analysis": learning_curve_analysis,
            "hyperparameter_sensitivity_analysis": hyperparameter_sensitivity_analysis,
            "ensemble_composition_analysis": ensemble_composition_analysis,
            "visualize_trees": visualize_trees,
            "max_trees_to_visualize": max_trees_to_visualize,
            "feature_importance_analysis": True,
            "prediction_distribution_analysis": True,
            "cross_validation_analysis": cross_validation_analysis,
            "cv_folds": cv_folds,
            "performance_benchmarking": performance_benchmarking
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return RandomForestRegressorPlugin(
            n_estimators=hyperparameters.get("n_estimators", self.n_estimators),
            criterion=hyperparameters.get("criterion", self.criterion),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            min_samples_split=hyperparameters.get("min_samples_split", self.min_samples_split),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", self.min_samples_leaf),
            min_weight_fraction_leaf=hyperparameters.get("min_weight_fraction_leaf", self.min_weight_fraction_leaf),
            max_features=hyperparameters.get("max_features", self.max_features),
            max_leaf_nodes=hyperparameters.get("max_leaf_nodes", self.max_leaf_nodes),
            min_impurity_decrease=hyperparameters.get("min_impurity_decrease", self.min_impurity_decrease),
            bootstrap=hyperparameters.get("bootstrap", self.bootstrap),
            oob_score=hyperparameters.get("oob_score", self.oob_score),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            random_state=hyperparameters.get("random_state", self.random_state),
            verbose=hyperparameters.get("verbose", self.verbose),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            ccp_alpha=hyperparameters.get("ccp_alpha", self.ccp_alpha),
            max_samples=hyperparameters.get("max_samples", self.max_samples),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            compute_permutation_importance=hyperparameters.get("compute_permutation_importance", self.compute_permutation_importance),
            compute_oob_analysis=hyperparameters.get("compute_oob_analysis", self.compute_oob_analysis),
            tree_diversity_analysis=hyperparameters.get("tree_diversity_analysis", self.tree_diversity_analysis),
            ensemble_convergence_analysis=hyperparameters.get("ensemble_convergence_analysis", self.ensemble_convergence_analysis),
            feature_interaction_analysis=hyperparameters.get("feature_interaction_analysis", self.feature_interaction_analysis),
            learning_curve_analysis=hyperparameters.get("learning_curve_analysis", self.learning_curve_analysis),
            hyperparameter_sensitivity_analysis=hyperparameters.get("hyperparameter_sensitivity_analysis", self.hyperparameter_sensitivity_analysis),
            ensemble_composition_analysis=hyperparameters.get("ensemble_composition_analysis", self.ensemble_composition_analysis),
            visualize_trees=hyperparameters.get("visualize_trees", self.visualize_trees),
            max_trees_to_visualize=hyperparameters.get("max_trees_to_visualize", self.max_trees_to_visualize),
            feature_importance_analysis=hyperparameters.get("feature_importance_analysis", self.feature_importance_analysis),
            prediction_distribution_analysis=hyperparameters.get("prediction_distribution_analysis", self.prediction_distribution_analysis),
            cross_validation_analysis=hyperparameters.get("cross_validation_analysis", self.cross_validation_analysis),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            performance_benchmarking=hyperparameters.get("performance_benchmarking", self.performance_benchmarking)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Random Forest (minimal preprocessing needed)"""
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
        """Check if Random Forest is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Random Forest requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Random Forest Regressor requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # Sample size assessment
            if n_samples >= 1000:
                advantages.append(f"Large sample size ({n_samples}) - excellent for ensemble diversity")
            elif n_samples >= 200:
                advantages.append(f"Good sample size ({n_samples}) - adequate for robust ensemble")
            else:
                considerations.append(f"Small sample size ({n_samples}) - ensemble may have limited diversity")
            
            # Feature dimensionality assessment
            if n_features <= 50:
                advantages.append(f"Moderate dimensionality ({n_features}) - good for feature randomness")
            elif n_features <= 200:
                advantages.append(f"High dimensionality ({n_features}) - benefits from feature randomness")
            else:
                considerations.append(f"Very high dimensionality ({n_features}) - consider feature selection")
            
            # Data characteristics
            advantages.append("Robust ensemble method with built-in regularization")
            advantages.append("Automatic feature importance ranking")
            advantages.append("Built-in cross-validation through OOB samples")
            advantages.append("Handles non-linear relationships and interactions")
            
            # Check ensemble suitability
            trees_per_feature = n_samples / n_features
            if trees_per_feature < 5:
                considerations.append(f"Low samples-to-features ratio ({trees_per_feature:.1f}) - ensemble diversity may be limited")
            elif trees_per_feature < 20:
                considerations.append(f"Moderate samples-to-features ratio ({trees_per_feature:.1f}) - good for ensemble")
            else:
                advantages.append(f"High samples-to-features ratio ({trees_per_feature:.1f}) - excellent for diverse ensemble")
            
            # Bootstrap suitability
            if n_samples >= 100:
                advantages.append("Sample size supports effective bootstrap sampling")
            else:
                considerations.append("Small sample size may limit bootstrap effectiveness")
            
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
                f"🌲 Suitability for Random Forest: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance with multiple methods"""
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
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, name in enumerate(self.feature_names_):
            feature_importance[name] = {
                'gini_importance': builtin_importance[i],
                'permutation_importance': permutation_importance[i] if permutation_importance is not None else None,
                'rank': np.where(feature_ranking == i)[0][0] + 1,
                'consensus': analysis['importance_stability']['consensus_importance'][i],
                'stability_cv': analysis['importance_stability']['cv_importance'][i],
                'is_stable_important': analysis['stable_important_features'][i]
            }
        
        # Get top features
        top_features = analysis['top_features']
        
        return {
            'feature_importance': feature_importance,
            'top_features': [(f['name'], f['builtin_importance'], f.get('permutation_importance')) for f in top_features],
            'importance_statistics': analysis['importance_statistics'],
            'ensemble_info': {
                'n_trees': len(self.model_.estimators_),
                'consensus_threshold': analysis['consensus_threshold'],
                'feature_stability': 'High' if np.mean(analysis['importance_stability']['cv_importance']) < 0.5 else 'Low'
            },
            'interpretation': 'Ensemble feature importance (Gini impurity decrease averaged across trees)'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "algorithm": "Random Forest Regressor",
            "ensemble_structure": {
                "n_estimators": len(self.model_.estimators_),
                "bootstrap": self.bootstrap,
                "max_features": self.max_features,
                "oob_score_available": hasattr(self.model_, 'oob_score_')
            },
            "tree_parameters": {
                "criterion": self.criterion,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "ccp_alpha": self.ccp_alpha
            },
            "ensemble_analysis": {
                "oob_performance": self.oob_analysis_.get('performance_assessment', {}),
                "tree_diversity": self.tree_diversity_analysis_.get('diversity_assessment', {}),
                "convergence": self.ensemble_convergence_analysis_.get('convergence_analysis', {})
            },
            "interpretability": {
                "feature_importance_available": True,
                "tree_structure_interpretable": True,
                "ensemble_uncertainty": True,
                "oob_validation": self.oob_score
            }
        }
    
    def get_ensemble_analysis(self) -> Dict[str, Any]:
        """Get comprehensive ensemble analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        return {
            "feature_importance_analysis": self.feature_importance_analysis_,
            "oob_analysis": self.oob_analysis_,
            "tree_diversity_analysis": self.tree_diversity_analysis_,
            "ensemble_convergence_analysis": self.ensemble_convergence_analysis_,
            "feature_interaction_analysis": self.feature_interaction_analysis_,
            "learning_curve_analysis": self.learning_curve_analysis_,
            "hyperparameter_sensitivity_analysis": self.hyperparameter_sensitivity_analysis_,
            "ensemble_composition_analysis": self.ensemble_composition_analysis_,
            "cross_validation_analysis": self.cross_validation_analysis_,
            "performance_benchmarking": self.performance_benchmarking_
        }
    
    def plot_ensemble_analysis(self, figsize=(20, 15)):
        """Plot comprehensive ensemble analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to plot ensemble analysis")
        
        fig, axes = plt.subplots(4, 4, figsize=figsize)
        axes = axes.ravel()
        
        # Plot 1: Feature importance
        ax1 = axes[0]
        if 'builtin_importance' in self.feature_importance_analysis_:
            importance = self.feature_importance_analysis_['builtin_importance']
            feature_names = self.feature_names_[:10]  # Top 10
            importance = importance[:10]
            
            y_pos = np.arange(len(feature_names))
            bars = ax1.barh(y_pos, importance, alpha=0.7, color='forestgreen')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([name[:15] for name in feature_names], fontsize=8)
            ax1.set_xlabel('Importance')
            ax1.set_title('Feature Importance (Gini)')
            ax1.grid(True, alpha=0.3)
            
            # Add consensus info if available
            if 'importance_stability' in self.feature_importance_analysis_:
                consensus = self.feature_importance_analysis_['importance_stability']['consensus_importance'][:10]
                for i, (bar, cons) in enumerate(zip(bars, consensus)):
                    ax1.text(bar.get_width() + max(importance) * 0.01, bar.get_y() + bar.get_height()/2,
                            f'{cons:.1f}', va='center', fontsize=8, alpha=0.7)
        else:
            ax1.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Feature Importance')
        
        # Plot 2: OOB Performance
        ax2 = axes[1]
        if 'oob_score_r2' in self.oob_analysis_:
            oob_data = self.oob_analysis_
            
            metrics = ['OOB R²', 'Train R²', 'OOB RMSE']
            values = [
                oob_data['oob_score_r2'],
                oob_data['training_performance']['train_r2'],
                oob_data['oob_rmse'] / max(oob_data['oob_rmse'], 1)  # Normalized
            ]
            colors = ['lightblue', 'orange', 'lightcoral']
            
            bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
            ax2.set_ylabel('Score')
            ax2.set_title('OOB Performance Analysis')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'OOB analysis\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('OOB Performance')
        
        # Plot 3: Tree Diversity
        ax3 = axes[2]
        if 'prediction_diversity' in self.tree_diversity_analysis_:
            diversity_data = self.tree_diversity_analysis_['prediction_diversity']
            
            metrics = ['Diversity\nScore', 'Mean\nCorrelation', 'Low Corr\nPairs']
            values = [
                diversity_data['diversity_score'],
                diversity_data['mean_correlation'],
                diversity_data['low_correlation_pairs']
            ]
            colors = ['lightgreen', 'lightyellow', 'lightpink']
            
            bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
            ax3.set_ylabel('Value')
            ax3.set_title('Tree Diversity Metrics')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Tree diversity\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Tree Diversity')
        
        # Plot 4: Ensemble Convergence
        ax4 = axes[3]
        if 'tree_counts' in self.ensemble_convergence_analysis_:
            conv_data = self.ensemble_convergence_analysis_
            tree_counts = conv_data['tree_counts']
            scores = conv_data['training_scores']
            
            ax4.plot(tree_counts, scores, 'o-', color='purple', alpha=0.7, linewidth=2)
            ax4.set_xlabel('Number of Trees')
            ax4.set_ylabel('R² Score')
            ax4.set_title('Ensemble Convergence')
            ax4.grid(True, alpha=0.3)
            
            # Mark optimal point if available
            if 'convergence_analysis' in conv_data and 'optimal_trees' in conv_data['convergence_analysis']:
                optimal = conv_data['convergence_analysis']['optimal_trees']
                if optimal in tree_counts:
                    idx = tree_counts.index(optimal)
                    ax4.axvline(x=optimal, color='red', linestyle='--', alpha=0.7,
                               label=f'Optimal: {optimal}')
                    ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Convergence analysis\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Ensemble Convergence')
        
        # Continue with remaining plots...
        # Plot 5: Cross-validation results
        ax5 = axes[4]
        if 'cv_results' in self.cross_validation_analysis_:
            cv_data = self.cross_validation_analysis_['cv_results']
            if 'r2' in cv_data:
                cv_scores = cv_data['r2']['scores']
                ax5.boxplot([cv_scores], labels=['CV R²'])
                ax5.scatter([1] * len(cv_scores), cv_scores, alpha=0.7, color='red', s=30)
                
                # Add training score line
                train_r2 = self.cross_validation_analysis_['training_performance']['train_r2']
                ax5.axhline(y=train_r2, color='blue', linestyle='--', 
                           label=f'Train R²: {train_r2:.3f}')
                ax5.legend()
                ax5.set_ylabel('R² Score')
                ax5.set_title('Cross-Validation Performance')
                ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'CV analysis\nnot available', 
                    ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Cross-Validation')
        
        # Add remaining empty plots with information
        for i in range(5, 16):
            ax = axes[i]
            if i == 5:
                ax.text(0.5, 0.5, 'Learning Curves\n(if enabled)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Learning Curves')
            elif i == 6:
                ax.text(0.5, 0.5, 'Feature Interactions\n(if enabled)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Interactions')
            elif i == 7:
                ax.text(0.5, 0.5, 'Hyperparameter\nSensitivity\n(if enabled)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Hyperparameter Sensitivity')
            else:
                # Create ensemble summary
                if i == 8:
                    try:
                        summary_text = f"""
Random Forest Summary:
• Trees: {len(self.model_.estimators_)}
• Bootstrap: {self.bootstrap}
• Max Features: {self.max_features}
• OOB R²: {self.oob_analysis_.get('oob_score_r2', 'N/A'):.3f}

Diversity: {self.tree_diversity_analysis_.get('diversity_assessment', {}).get('diversity_level', 'Unknown')}

Performance: {self.oob_analysis_.get('performance_assessment', {}).get('overall_assessment', 'Unknown')}
"""
                        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=9,
                               verticalalignment='top', fontfamily='monospace')
                        ax.set_title('Ensemble Summary')
                    except:
                        ax.text(0.5, 0.5, 'Summary not\navailable', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('Ensemble Summary')
                else:
                    ax.text(0.5, 0.5, f'Analysis Plot {i-8}\n(available based on\nconfiguration)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Additional Analysis {i-8}')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Random Forest Regressor",
            "type": "Ensemble method using bootstrap aggregation of decision trees",
            "training_completed": True,
            "ensemble_characteristics": {
                "robust_ensemble": True,
                "built_in_regularization": True,
                "feature_importance_available": True,
                "uncertainty_quantification": True,
                "built_in_cross_validation": self.oob_score,
                "parallel_training": True
            },
            "ensemble_structure": {
                "n_estimators": len(self.model_.estimators_),
                "bootstrap_sampling": self.bootstrap,
                "feature_randomness": self.max_features,
                "oob_samples_used": self.oob_score
            },
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "criterion": self.criterion,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "bootstrap": self.bootstrap
            }
        }
        
        # Add analysis information if available
        if self.oob_analysis_:
            info["oob_performance"] = {
                "oob_r2": self.oob_analysis_.get('oob_score_r2'),
                "performance_assessment": self.oob_analysis_.get('performance_assessment', {}).get('overall_assessment'),
                "overfitting_assessment": self.oob_analysis_.get('performance_assessment', {}).get('overfitting_assessment')
            }
        
        if self.tree_diversity_analysis_:
            info["ensemble_diversity"] = {
                "diversity_level": self.tree_diversity_analysis_.get('diversity_assessment', {}).get('diversity_level'),
                "diversity_score": self.tree_diversity_analysis_.get('prediction_diversity', {}).get('diversity_score'),
                "mean_correlation": self.tree_diversity_analysis_.get('prediction_diversity', {}).get('mean_correlation')
            }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Random Forest Regressor model.

        These metrics are derived from the comprehensive analyses performed during fit().
        Parameters y_true, y_pred, y_proba are kept for API consistency but are not
        directly used as metrics are sourced from internal analysis attributes.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values.
        y_pred : np.ndarray, optional
            Predicted target values.
        y_proba : np.ndarray, optional
            Predicted probabilities or uncertainty measures (not standard for regressors).

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve Random Forest Regressor specific metrics."}

        metrics = {}
        prefix = "rfr_" # Prefix for Random Forest Regressor specific metrics

        # OOB Performance Metrics
        if hasattr(self, 'oob_analysis_') and self.oob_analysis_ and not self.oob_analysis_.get('error'):
            if 'oob_score_r2' in self.oob_analysis_:
                metrics[f"{prefix}oob_score_r2"] = self.oob_analysis_['oob_score_r2']
            if 'overfitting_metrics' in self.oob_analysis_ and 'r2_gap' in self.oob_analysis_['overfitting_metrics']:
                metrics[f"{prefix}oob_overfitting_r2_gap"] = self.oob_analysis_['overfitting_metrics']['r2_gap']
            if 'performance_assessment' in self.oob_analysis_ and 'generalization_score' in self.oob_analysis_['performance_assessment']:
                 metrics[f"{prefix}oob_generalization_score"] = self.oob_analysis_['performance_assessment']['generalization_score']
            if 'oob_rmse' in self.oob_analysis_:
                metrics[f"{prefix}oob_rmse"] = self.oob_analysis_['oob_rmse']


        # Feature Importance Metrics
        if hasattr(self, 'feature_importance_analysis_') and self.feature_importance_analysis_ and not self.feature_importance_analysis_.get('error'):
            fi_stats = self.feature_importance_analysis_.get('importance_statistics', {})
            if 'gini_coefficient' in fi_stats:
                metrics[f"{prefix}feature_importance_gini_coeff"] = fi_stats['gini_coefficient']
            if 'importance_concentration' in fi_stats and 'top_5_concentration' in fi_stats['importance_concentration']:
                metrics[f"{prefix}feature_importance_concentration_top_5_pct"] = fi_stats['importance_concentration']['top_5_concentration'] * 100
            if 'mean_importance' in fi_stats:
                 metrics[f"{prefix}mean_builtin_feature_importance"] = fi_stats['mean_importance']
            if 'stable_important_features' in self.feature_importance_analysis_:
                 metrics[f"{prefix}num_stable_important_features"] = int(np.sum(self.feature_importance_analysis_['stable_important_features']))


        # Tree Diversity Metrics
        if hasattr(self, 'tree_diversity_analysis_') and self.tree_diversity_analysis_ and not self.tree_diversity_analysis_.get('error'):
            pred_div = self.tree_diversity_analysis_.get('prediction_diversity', {})
            struct_div = self.tree_diversity_analysis_.get('structure_diversity', {})
            if 'diversity_score' in pred_div:
                metrics[f"{prefix}tree_prediction_diversity_score"] = pred_div['diversity_score']
            if 'mean_correlation' in pred_div:
                metrics[f"{prefix}mean_tree_correlation"] = pred_div['mean_correlation']
            
            if 'depth_diversity' in struct_div and 'mean' in struct_div['depth_diversity']:
                metrics[f"{prefix}mean_tree_depth"] = struct_div['depth_diversity']['mean']
            if 'depth_diversity' in struct_div and 'std' in struct_div['depth_diversity']:
                metrics[f"{prefix}std_tree_depth"] = struct_div['depth_diversity']['std']

        # Ensemble Convergence Metrics
        if hasattr(self, 'ensemble_convergence_analysis_') and self.ensemble_convergence_analysis_ and not self.ensemble_convergence_analysis_.get('error'):
            conv_analysis = self.ensemble_convergence_analysis_.get('convergence_analysis', {})
            if 'optimal_trees' in conv_analysis:
                metrics[f"{prefix}ensemble_convergence_optimal_trees"] = conv_analysis['optimal_trees']
            if 'final_score' in conv_analysis:
                metrics[f"{prefix}ensemble_convergence_final_r2_score"] = conv_analysis['final_score'] # Assuming R2 is the score
            if 'score_stability' in conv_analysis:
                metrics[f"{prefix}ensemble_convergence_score_stability"] = conv_analysis['score_stability']

        # Cross-Validation Metrics
        if hasattr(self, 'cross_validation_analysis_') and self.cross_validation_analysis_ and not self.cross_validation_analysis_.get('error'):
            cv_res = self.cross_validation_analysis_.get('cv_results', {})
            perf_assess = self.cross_validation_analysis_.get('performance_assessment', {})
            if 'r2' in cv_res and 'mean' in cv_res['r2']:
                 metrics[f"{prefix}mean_cv_r2"] = cv_res['r2']['mean']
            if 'generalization_gap' in perf_assess:
                 metrics[f"{prefix}cv_generalization_gap_r2"] = perf_assess['generalization_gap']
            if 'neg_mean_squared_error' in cv_res and 'mean' in cv_res['neg_mean_squared_error']:
                 metrics[f"{prefix}mean_cv_neg_mse"] = cv_res['neg_mean_squared_error']['mean']


        # Prediction Uncertainty from predict_with_uncertainty (if y_pred and y_proba are structured for it)
        # This part is speculative as it depends on how y_pred/y_proba might be used.
        # For now, we rely on the internal analyses.
        # Example: if y_proba contained std_devs for each prediction in y_pred:
        # if y_pred is not None and y_proba is not None and len(y_pred) == len(y_proba):
        #     try:
        #         # Assuming y_proba contains standard deviations of predictions
        #         avg_prediction_std_dev = np.mean(y_proba)
        #         metrics[f"{prefix}avg_prediction_std_dev"] = float(avg_prediction_std_dev)
        #         # Coefficient of variation (if y_pred is not zero)
        #         y_pred_abs_mean = np.mean(np.abs(y_pred))
        #         if y_pred_abs_mean > 1e-9:
        #             metrics[f"{prefix}avg_prediction_coeff_variation"] = float(avg_prediction_std_dev / y_pred_abs_mean)
        #     except Exception:
        #         pass # Could not calculate uncertainty from y_pred/y_proba

        if not metrics:
            metrics['info'] = "No specific Random Forest Regressor metrics were available or calculated from internal analyses."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return RandomForestRegressorPlugin()
