import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.ensemble import ExtraTreesRegressor
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


class ExtraTreesRegressorPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Extra Trees Regressor Plugin - Extremely Randomized Trees
    
    Extra Trees (Extremely Randomized Trees) is an advanced ensemble method that extends
    Random Forest by introducing additional randomness in the tree building process.
    It uses random thresholds for each feature rather than searching for the best possible
    thresholds, leading to faster training and often better generalization.
    
    Key Features:
    - Extreme randomization: Random splits at each node
    - Faster training than Random Forest
    - Often better bias-variance tradeoff
    - Robust ensemble with built-in regularization
    - Excellent for high-dimensional data
    - Superior generalization capabilities
    - Built-in feature importance with stability analysis
    - Comprehensive ensemble diversity metrics
    - Advanced split randomness analysis
    - Extensive computational efficiency benchmarking
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
        
        # Extra Trees specific parameters
        max_features='sqrt',  # Key parameter for Extra Trees
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,  # Default False for Extra Trees
        
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
        split_randomness_analysis=True,
        tree_diversity_analysis=True,
        ensemble_convergence_analysis=True,
        
        # Advanced analysis
        feature_interaction_analysis=True,
        learning_curve_analysis=True,
        hyperparameter_sensitivity_analysis=True,
        computational_efficiency_analysis=True,
        
        # Comparison with Random Forest
        compare_with_random_forest=True,
        randomness_impact_analysis=True,
        
        # Visualization options
        visualize_trees=False,
        max_trees_to_visualize=3,
        feature_importance_analysis=True,
        prediction_distribution_analysis=True,
        
        # Performance analysis
        cross_validation_analysis=True,
        cv_folds=5,
        performance_benchmarking=True,
        
        # Extra Trees specific analysis
        split_quality_analysis=True,
        randomness_efficiency_analysis=True,
        bias_variance_analysis=True
    ):
        super().__init__()
        
        # Core ensemble parameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        
        # Extra Trees specific parameters
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        
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
        self.split_randomness_analysis = split_randomness_analysis
        self.tree_diversity_analysis = tree_diversity_analysis
        self.ensemble_convergence_analysis = ensemble_convergence_analysis
        
        # Advanced analysis
        self.feature_interaction_analysis = feature_interaction_analysis
        self.learning_curve_analysis = learning_curve_analysis
        self.hyperparameter_sensitivity_analysis = hyperparameter_sensitivity_analysis
        self.computational_efficiency_analysis = computational_efficiency_analysis
        
        # Comparison analysis
        self.compare_with_random_forest = compare_with_random_forest
        self.randomness_impact_analysis = randomness_impact_analysis
        
        # Visualization options
        self.visualize_trees = visualize_trees
        self.max_trees_to_visualize = max_trees_to_visualize
        self.feature_importance_analysis = feature_importance_analysis
        self.prediction_distribution_analysis = prediction_distribution_analysis
        
        # Performance analysis
        self.cross_validation_analysis = cross_validation_analysis
        self.cv_folds = cv_folds
        self.performance_benchmarking = performance_benchmarking
        
        # Extra Trees specific analysis
        self.split_quality_analysis = split_quality_analysis
        self.randomness_efficiency_analysis = randomness_efficiency_analysis
        self.bias_variance_analysis = bias_variance_analysis
        
        # Required plugin metadata
        self._name = "Extra Trees Regressor"
        self._description = "Extremely randomized trees with enhanced randomness for superior generalization"
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
        self.split_randomness_analysis_ = {}
        self.tree_diversity_analysis_ = {}
        self.ensemble_convergence_analysis_ = {}
        self.feature_interaction_analysis_ = {}
        self.learning_curve_analysis_ = {}
        self.hyperparameter_sensitivity_analysis_ = {}
        self.computational_efficiency_analysis_ = {}
        self.random_forest_comparison_ = {}
        self.randomness_impact_analysis_ = {}
        self.cross_validation_analysis_ = {}
        self.performance_benchmarking_ = {}
        self.split_quality_analysis_ = {}
        self.randomness_efficiency_analysis_ = {}
        self.bias_variance_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Extra Trees Regressor with comprehensive analysis
        
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
        
        # Create and configure Extra Trees model
        self.model_ = ExtraTreesRegressor(
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
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples
        )
        
        # Fit the model
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        # Perform comprehensive analysis
        self._analyze_feature_importance()
        self._analyze_split_randomness()
        self._analyze_tree_diversity()
        self._analyze_ensemble_convergence()
        self._analyze_feature_interactions()
        self._analyze_learning_curves()
        self._analyze_hyperparameter_sensitivity()
        self._analyze_computational_efficiency()
        self._compare_with_random_forest()
        self._analyze_randomness_impact()
        self._analyze_cross_validation()
        self._analyze_performance_benchmarks()
        self._analyze_split_quality()
        self._analyze_randomness_efficiency()
        self._analyze_bias_variance()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted Extra Trees
        
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
        
        # Extra Trees specific: Calculate randomness-based uncertainty
        median_predictions = np.median(tree_predictions, axis=0)
        mad_predictions = np.median(np.abs(tree_predictions - median_predictions), axis=0)
        
        return {
            'predictions': mean_predictions,
            'std_predictions': std_predictions,
            'confidence_95_lower': confidence_95_lower,
            'confidence_95_upper': confidence_95_upper,
            'prediction_interval_width': confidence_95_upper - confidence_95_lower,
            'coefficient_of_variation': cv,
            'tree_predictions': tree_predictions,
            'ensemble_agreement': 1.0 / (1.0 + cv),
            'uncertainty_score': cv,
            'median_predictions': median_predictions,
            'mad_predictions': mad_predictions,
            'randomness_uncertainty': mad_predictions / (np.abs(median_predictions) + 1e-10)
        }
    
    def _analyze_feature_importance(self):
        """Analyze feature importance with Extra Trees specific insights"""
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
                'consensus_importance': np.mean(tree_importances > 0, axis=0),
                'randomness_robustness': 1.0 / (1.0 + np.std(tree_importances, axis=0))
            }
            
            # Extra Trees specific: Analyze random split impact on importance
            random_split_impact = self._analyze_random_split_importance_impact(tree_importances)
            
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
                'randomness_stability': np.mean(importance_stability['randomness_robustness'])
            }
            
            # Identify robust important features (stable under extreme randomness)
            stable_threshold = np.mean(builtin_importance) + 0.5 * np.std(builtin_importance)
            robustness_threshold = 0.6  # High robustness to random splits
            
            robust_important_features = (
                (builtin_importance > stable_threshold) & 
                (importance_stability['randomness_robustness'] > robustness_threshold)
            )
            
            self.feature_importance_analysis_ = {
                'builtin_importance': builtin_importance,
                'permutation_importance': permutation_imp,
                'permutation_importance_std': permutation_imp_std,
                'importance_stability': importance_stability,
                'random_split_impact': random_split_impact,
                'feature_ranking': importance_ranking,
                'feature_names': self.feature_names_,
                'importance_statistics': importance_stats,
                'robust_important_features': robust_important_features,
                'robustness_threshold': robustness_threshold,
                'stable_threshold': stable_threshold,
                'top_features': [
                    {
                        'name': self.feature_names_[i],
                        'builtin_importance': builtin_importance[i],
                        'permutation_importance': permutation_imp[i] if permutation_imp is not None else None,
                        'importance_std': importance_stability['std_importance'][i],
                        'consensus': importance_stability['consensus_importance'][i],
                        'randomness_robustness': importance_stability['randomness_robustness'][i],
                        'stability_cv': importance_stability['cv_importance'][i]
                    }
                    for i in importance_ranking[:15]
                ]
            }
            
        except Exception as e:
            self.feature_importance_analysis_ = {
                'error': f'Could not analyze feature importance: {str(e)}'
            }
    
    def _analyze_random_split_importance_impact(self, tree_importances):
        """Analyze how random splits affect feature importance"""
        try:
            n_trees, n_features = tree_importances.shape
            
            # Calculate variance in importance across trees for each feature
            importance_variance = np.var(tree_importances, axis=0)
            importance_mean = np.mean(tree_importances, axis=0)
            
            # Coefficient of variation for each feature
            cv_per_feature = importance_variance / (importance_mean + 1e-10)
            
            # Features with high variance indicate sensitivity to random splits
            high_variance_features = cv_per_feature > np.percentile(cv_per_feature, 75)
            low_variance_features = cv_per_feature < np.percentile(cv_per_feature, 25)
            
            return {
                'importance_variance_per_feature': importance_variance,
                'cv_per_feature': cv_per_feature,
                'high_variance_features': high_variance_features,
                'low_variance_features': low_variance_features,
                'mean_cv': np.mean(cv_per_feature),
                'randomness_sensitivity_score': np.mean(cv_per_feature),
                'stable_feature_ratio': np.mean(low_variance_features),
                'interpretation': 'Lower CV indicates features robust to random splits'
            }
            
        except:
            return {'error': 'Could not analyze random split impact'}
    
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
    
    def _analyze_split_randomness(self):
        """Analyze the impact of random splits in Extra Trees"""
        if not self.split_randomness_analysis:
            return
        
        try:
            # Analyze split characteristics across trees
            split_characteristics = []
            
            for tree in self.model_.estimators_[:10]:  # Sample trees for analysis
                tree_structure = tree.tree_
                
                # Extract split information
                features_used = tree_structure.feature[tree_structure.feature >= 0]
                thresholds_used = tree_structure.threshold[tree_structure.feature >= 0]
                
                split_characteristics.append({
                    'n_splits': len(features_used),
                    'unique_features': len(np.unique(features_used)),
                    'feature_diversity': len(np.unique(features_used)) / len(features_used) if len(features_used) > 0 else 0,
                    'threshold_variance': np.var(thresholds_used) if len(thresholds_used) > 0 else 0
                })
            
            # Aggregate split randomness metrics
            randomness_metrics = {
                'mean_splits_per_tree': np.mean([sc['n_splits'] for sc in split_characteristics]),
                'mean_unique_features_per_tree': np.mean([sc['unique_features'] for sc in split_characteristics]),
                'mean_feature_diversity': np.mean([sc['feature_diversity'] for sc in split_characteristics]),
                'mean_threshold_variance': np.mean([sc['threshold_variance'] for sc in split_characteristics]),
                'split_randomness_score': np.mean([sc['feature_diversity'] for sc in split_characteristics])
            }
            
            # Compare feature usage across trees
            all_features_used = []
            for tree in self.model_.estimators_:
                features_in_tree = np.unique(tree.tree_.feature[tree.tree_.feature >= 0])
                all_features_used.append(features_in_tree)
            
            # Feature usage frequency across ensemble
            feature_usage_freq = np.zeros(self.n_features_in_)
            for features in all_features_used:
                feature_usage_freq[features] += 1
            
            feature_usage_freq = feature_usage_freq / len(self.model_.estimators_)
            
            # Randomness assessment
            randomness_assessment = self._assess_split_randomness(
                randomness_metrics, feature_usage_freq
            )
            
            self.split_randomness_analysis_ = {
                'randomness_metrics': randomness_metrics,
                'feature_usage_frequency': feature_usage_freq,
                'randomness_assessment': randomness_assessment,
                'split_characteristics': split_characteristics,
                'trees_analyzed': min(10, len(self.model_.estimators_))
            }
            
        except Exception as e:
            self.split_randomness_analysis_ = {
                'error': f'Could not analyze split randomness: {str(e)}'
            }
    
    def _assess_split_randomness(self, randomness_metrics, feature_usage_freq):
        """Assess the quality of split randomness"""
        try:
            diversity_score = randomness_metrics['mean_feature_diversity']
            usage_uniformity = 1.0 - np.std(feature_usage_freq)
            
            # Randomness categories
            if diversity_score > 0.8 and usage_uniformity > 0.7:
                randomness_level = "Excellent - High diversity with uniform feature usage"
            elif diversity_score > 0.6 and usage_uniformity > 0.5:
                randomness_level = "Good - Solid randomness with balanced features"
            elif diversity_score > 0.4:
                randomness_level = "Moderate - Acceptable randomness level"
            else:
                randomness_level = "Low - Limited randomness, may reduce Extra Trees benefits"
            
            # Recommendations
            recommendations = []
            if diversity_score < 0.5:
                recommendations.extend([
                    "Consider increasing max_features parameter",
                    "Reduce min_samples_split for more diverse splits",
                    "Check if dataset allows for sufficient randomness"
                ])
            elif usage_uniformity < 0.5:
                recommendations.append("Some features are overused - verify feature scaling")
            else:
                recommendations.append("Split randomness is optimal for Extra Trees")
            
            return {
                'randomness_level': randomness_level,
                'diversity_score': diversity_score,
                'usage_uniformity': usage_uniformity,
                'overall_randomness_score': (diversity_score + usage_uniformity) / 2,
                'recommendations': recommendations
            }
            
        except:
            return {'randomness_level': 'Unknown', 'recommendations': []}
    
    def _analyze_tree_diversity(self):
        """Analyze diversity among trees with Extra Trees specific metrics"""
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
            
            for i in range(min(n_trees, 50)):
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
                'diversity_score': 1 - np.mean(correlations),
                'low_correlation_pairs': np.sum(correlations < 0.3) / len(correlations),  # Lower threshold for Extra Trees
                'negative_correlation_pairs': np.sum(correlations < 0) / len(correlations)
            }
            
            # Extra Trees specific: Analyze extreme diversity
            extreme_diversity_metrics = {
                'very_low_correlation_pairs': np.sum(correlations < 0.1) / len(correlations),
                'correlation_range': np.max(correlations) - np.min(correlations),
                'correlation_variance': np.var(correlations),
                'extreme_randomness_score': diversity_metrics['diversity_score'] * (1 + diversity_metrics['negative_correlation_pairs'])
            }
            
            # Analyze tree structure diversity
            tree_depths = [tree.tree_.max_depth for tree in self.model_.estimators_]
            tree_nodes = [tree.tree_.node_count for tree in self.model_.estimators_]
            tree_leaves = [tree.tree_.n_leaves for tree in self.model_.estimators_]
            
            structure_diversity = {
                'depth_diversity': {
                    'mean': np.mean(tree_depths),
                    'std': np.std(tree_depths),
                    'range': np.max(tree_depths) - np.min(tree_depths),
                    'cv': np.std(tree_depths) / np.mean(tree_depths)
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
            diversity_assessment = self._assess_ensemble_diversity(
                diversity_metrics, structure_diversity, extreme_diversity_metrics
            )
            
            self.tree_diversity_analysis_ = {
                'prediction_diversity': diversity_metrics,
                'extreme_diversity_metrics': extreme_diversity_metrics,
                'structure_diversity': structure_diversity,
                'diversity_assessment': diversity_assessment,
                'n_trees_analyzed': min(n_trees, 50),
                'n_samples_used': sample_size
            }
            
        except Exception as e:
            self.tree_diversity_analysis_ = {
                'error': f'Could not analyze tree diversity: {str(e)}'
            }
    
    def _assess_ensemble_diversity(self, diversity_metrics, structure_diversity, extreme_diversity_metrics):
        """Assess the diversity of the Extra Trees ensemble"""
        try:
            diversity_score = diversity_metrics['diversity_score']
            extreme_randomness_score = extreme_diversity_metrics['extreme_randomness_score']
            mean_correlation = diversity_metrics['mean_correlation']
            
            # Extra Trees should have higher diversity than Random Forest
            if diversity_score > 0.8:
                diversity_level = "Exceptional - Extreme randomness achieving superior diversity"
            elif diversity_score > 0.7:
                diversity_level = "Excellent - High diversity typical of Extra Trees"
            elif diversity_score > 0.5:
                diversity_level = "Good - Solid diversity benefits"
            elif diversity_score > 0.3:
                diversity_level = "Moderate - Limited Extra Trees advantage"
            else:
                diversity_level = "Poor - Random splits not providing sufficient diversity"
            
            # Recommendations specific to Extra Trees
            recommendations = []
            if diversity_score < 0.5:
                recommendations.extend([
                    "Increase max_features for more random feature selection",
                    "Reduce min_samples_split to allow more diverse splits",
                    "Consider increasing dataset size for better randomness",
                    "Verify that features have sufficient variance for random splits"
                ])
            elif diversity_score < 0.7:
                recommendations.append("Good diversity, consider fine-tuning for marginal improvements")
            else:
                recommendations.append("Excellent diversity - Extra Trees randomness is working optimally")
            
            return {
                'diversity_level': diversity_level,
                'diversity_score': diversity_score,
                'extreme_randomness_score': extreme_randomness_score,
                'mean_correlation': mean_correlation,
                'recommendations': recommendations,
                'extra_trees_advantage': 'High' if diversity_score > 0.7 else 'Medium' if diversity_score > 0.5 else 'Low',
                'optimal_range': 'Diversity score > 0.7 is excellent for Extra Trees'
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
                    
                except:
                    convergence_scores.append(np.nan)
            
            # Analyze convergence patterns
            convergence_analysis = self._analyze_convergence_pattern(tree_counts, convergence_scores)
            
            # Extra Trees specific: Analyze fast convergence due to extreme randomness
            fast_convergence_analysis = self._analyze_extra_trees_convergence(tree_counts, convergence_scores)
            
            self.ensemble_convergence_analysis_ = {
                'tree_counts': tree_counts,
                'training_scores': convergence_scores,
                'convergence_analysis': convergence_analysis,
                'fast_convergence_analysis': fast_convergence_analysis,
                'optimal_n_estimators': convergence_analysis.get('optimal_trees', self.n_estimators)
            }
            
        except Exception as e:
            self.ensemble_convergence_analysis_ = {
                'error': f'Could not analyze ensemble convergence: {str(e)}'
            }
    
    def _analyze_extra_trees_convergence(self, tree_counts, scores):
        """Analyze Extra Trees specific convergence characteristics"""
        try:
            valid_scores = [(t, s) for t, s in zip(tree_counts, scores) if not np.isnan(s)]
            
            if len(valid_scores) < 5:
                return {'status': 'Insufficient data for Extra Trees convergence analysis'}
            
            tree_counts_valid, scores_valid = zip(*valid_scores)
            scores_valid = np.array(scores_valid)
            
            # Analyze early convergence (Extra Trees often converge faster)
            early_scores = scores_valid[:len(scores_valid)//2]
            late_scores = scores_valid[len(scores_valid)//2:]
            
            early_improvement = np.mean(np.diff(early_scores)) if len(early_scores) > 1 else 0
            late_improvement = np.mean(np.diff(late_scores)) if len(late_scores) > 1 else 0
            
            # Fast convergence ratio
            fast_convergence_ratio = early_improvement / (late_improvement + 1e-10)
            
            # Convergence efficiency
            halfway_score = scores_valid[len(scores_valid)//2]
            final_score = scores_valid[-1]
            convergence_efficiency = halfway_score / final_score if final_score > 0 else 0
            
            return {
                'fast_convergence_ratio': fast_convergence_ratio,
                'convergence_efficiency': convergence_efficiency,
                'early_improvement_rate': early_improvement,
                'late_improvement_rate': late_improvement,
                'converges_fast': fast_convergence_ratio > 2.0,
                'efficiency_score': convergence_efficiency,
                'interpretation': 'Higher efficiency indicates faster convergence due to extreme randomness'
            }
            
        except:
            return {'status': 'Error in Extra Trees convergence analysis'}
    
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
            elif diminishing_point < current_trees * 0.4:
                return f"Extra Trees converge very fast - {diminishing_point} trees may be sufficient"
            elif diminishing_point < current_trees * 0.7:
                return f"Good convergence at ~{diminishing_point} trees. Current setting is reasonable"
            else:
                return f"Performance still improving. Current setting of {current_trees} trees is appropriate"
        except:
            return "Could not generate convergence recommendation"
    
    def _analyze_feature_interactions(self):
        """Analyze feature interactions in the Extra Trees ensemble"""
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
                        # Extra Trees specific: interaction analysis with random splits
                        interaction_strength = self._calculate_interaction_strength(
                            X_sample, feat1, feat2
                        )
                        
                        # Additional analysis for random split impact
                        random_split_interaction = self._analyze_random_split_interaction(
                            X_sample, feat1, feat2
                        )
                        
                        interactions.append({
                            'feature1': self.feature_names_[feat1],
                            'feature2': self.feature_names_[feat2],
                            'feature1_idx': feat1,
                            'feature2_idx': feat2,
                            'interaction_strength': interaction_strength,
                            'random_split_robustness': random_split_interaction,
                            'feature1_importance': self.model_.feature_importances_[feat1],
                            'feature2_importance': self.model_.feature_importances_[feat2]
                        })
                    except:
                        continue
            
            # Sort by interaction strength
            interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
            
            # Extra Trees specific: Analyze interaction stability under random splits
            interaction_stability = self._analyze_interaction_stability(interactions)
            
            self.feature_interaction_analysis_ = {
                'interactions': interactions[:10],  # Top 10 interactions
                'interaction_stability': interaction_stability,
                'n_features_analyzed': len(top_features),
                'sample_size_used': sample_size,
                'top_interactions': [
                    f"{inter['feature1']} × {inter['feature2']} (strength: {inter['interaction_strength']:.3f}, robustness: {inter['random_split_robustness']:.3f})"
                    for inter in interactions[:5]
                ]
            }
            
        except Exception as e:
            self.feature_interaction_analysis_ = {
                'error': f'Could not analyze feature interactions: {str(e)}'
            }
    
    def _analyze_random_split_interaction(self, X_sample, feat1_idx, feat2_idx):
        """Analyze how random splits affect feature interactions"""
        try:
            # Get subset of trees for analysis
            sample_trees = self.model_.estimators_[:10]
            
            interaction_strengths = []
            
            for tree in sample_trees:
                # Calculate interaction for individual tree
                baseline_pred = tree.predict(X_sample)
                
                # Perturb features and calculate interaction
                X_perturb1 = X_sample.copy()
                X_perturb1[:, feat1_idx] = np.median(X_sample[:, feat1_idx])
                pred1 = tree.predict(X_perturb1)
                
                X_perturb2 = X_sample.copy()
                X_perturb2[:, feat2_idx] = np.median(X_sample[:, feat2_idx])
                pred2 = tree.predict(X_perturb2)
                
                X_perturb_both = X_sample.copy()
                X_perturb_both[:, feat1_idx] = np.median(X_sample[:, feat1_idx])
                X_perturb_both[:, feat2_idx] = np.median(X_sample[:, feat2_idx])
                pred_both = tree.predict(X_perturb_both)
                
                # Calculate interaction
                additive_effect = (pred1 - baseline_pred) + (pred2 - baseline_pred)
                combined_effect = pred_both - baseline_pred
                interaction = np.mean(np.abs(combined_effect - additive_effect))
                
                interaction_strengths.append(interaction)
            
            # Calculate robustness to random splits
            interaction_robustness = 1.0 / (1.0 + np.std(interaction_strengths))
            
            return interaction_robustness
            
        except:
            return 0.0
    
    def _analyze_interaction_stability(self, interactions):
        """Analyze stability of interactions under random splits"""
        try:
            if not interactions:
                return {'stability_score': 0.0}
            
            robustness_scores = [inter.get('random_split_robustness', 0) for inter in interactions]
            
            stability_metrics = {
                'mean_robustness': np.mean(robustness_scores),
                'std_robustness': np.std(robustness_scores),
                'min_robustness': np.min(robustness_scores),
                'max_robustness': np.max(robustness_scores),
                'stable_interactions_ratio': np.mean(np.array(robustness_scores) > 0.5)
            }
            
            # Overall stability assessment
            if stability_metrics['mean_robustness'] > 0.7:
                stability_level = "High - Interactions are robust to random splits"
            elif stability_metrics['mean_robustness'] > 0.5:
                stability_level = "Medium - Moderately stable interactions"
            else:
                stability_level = "Low - Interactions sensitive to random splits"
            
            return {
                'stability_metrics': stability_metrics,
                'stability_level': stability_level,
                'interpretation': 'Higher robustness indicates interactions detectable despite random splits'
            }
            
        except:
            return {'stability_score': 0.0, 'error': 'Could not analyze stability'}
    
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
            simple_model = ExtraTreesRegressor(
                n_estimators=min(50, self.n_estimators),
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=1,
                bootstrap=self.bootstrap
            )
            
            # Calculate learning curves
            train_sizes_abs, train_scores, val_scores = learning_curve(
                simple_model, self.X_original_, self.y_original_,
                train_sizes=train_sizes,
                cv=3,
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
            
            # Extra Trees specific: Analyze fast learning due to randomness
            fast_learning_analysis = self._analyze_extra_trees_learning(
                train_sizes_abs, train_scores_mean, val_scores_mean
            )
            
            self.learning_curve_analysis_ = {
                'train_sizes': train_sizes_abs,
                'train_scores_mean': train_scores_mean,
                'train_scores_std': train_scores_std,
                'val_scores_mean': val_scores_mean,
                'val_scores_std': val_scores_std,
                'learning_analysis': learning_analysis,
                'fast_learning_analysis': fast_learning_analysis
            }
            
        except Exception as e:
            self.learning_curve_analysis_ = {
                'error': f'Could not analyze learning curves: {str(e)}'
            }
    
    def _analyze_extra_trees_learning(self, train_sizes, train_scores, val_scores):
        """Analyze Extra Trees specific learning characteristics"""
        try:
            # Analyze early vs late learning efficiency
            mid_point = len(train_sizes) // 2
            early_val_improvement = val_scores[mid_point] - val_scores[0] if len(val_scores) > mid_point else 0
            late_val_improvement = val_scores[-1] - val_scores[mid_point] if len(val_scores) > mid_point else 0
            
            # Learning efficiency ratio
            learning_efficiency = early_val_improvement / (late_val_improvement + 1e-10)
            
            # Data efficiency
            small_data_performance = val_scores[2] if len(val_scores) > 2 else 0  # ~30% of data
            full_data_performance = val_scores[-1]
            data_efficiency = small_data_performance / full_data_performance if full_data_performance > 0 else 0
            
            return {
                'learning_efficiency_ratio': learning_efficiency,
                'data_efficiency': data_efficiency,
                'early_learning_dominant': learning_efficiency > 2.0,
                'small_data_effective': data_efficiency > 0.8,
                'interpretation': 'Extra Trees often learn effectively with smaller datasets due to extreme randomness'
            }
            
        except:
            return {'status': 'Error in Extra Trees learning analysis'}
    
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
            
            # Analyze max_features sensitivity (crucial for Extra Trees)
            feature_values = ['sqrt', 'log2', None, 0.5] if self.max_features != 'sqrt' else ['sqrt', 'log2', None]
            feature_sensitivity = self._test_hyperparameter_values(
                'max_features', feature_values, max_tests=3
            )
            sensitivity_results['max_features'] = feature_sensitivity
            
            # Analyze max_depth sensitivity
            depth_values = [5, 10, None] if self.max_depth is None else [5, self.max_depth, None]
            depth_sensitivity = self._test_hyperparameter_values(
                'max_depth', depth_values, max_tests=3
            )
            sensitivity_results['max_depth'] = depth_sensitivity
            
            # Analyze bootstrap sensitivity (Extra Trees specific)
            bootstrap_values = [True, False]
            bootstrap_sensitivity = self._test_hyperparameter_values(
                'bootstrap', bootstrap_values, max_tests=2
            )
            sensitivity_results['bootstrap'] = bootstrap_sensitivity
            
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
                        'n_estimators': min(30, self.n_estimators),
                        'random_state': self.random_state,
                        'n_jobs': 1,
                        'bootstrap': self.bootstrap
                    }
                    params[param_name] = value
                    
                    model = ExtraTreesRegressor(**params)
                    
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
        """Assess overall hyperparameter sensitivity for Extra Trees"""
        try:
            sensitive_params = []
            stable_params = []
            
            for param, result in sensitivity_results.items():
                if 'score_range' in result:
                    if result['score_range'] > 0.1:
                        sensitive_params.append(param)
                    else:
                        stable_params.append(param)
            
            # Extra Trees specific assessment
            extra_trees_notes = []
            if 'max_features' in sensitive_params:
                extra_trees_notes.append("max_features significantly impacts randomness")
            if 'bootstrap' in sensitive_params:
                extra_trees_notes.append("bootstrap choice affects ensemble diversity")
            
            return {
                'sensitive_parameters': sensitive_params,
                'stable_parameters': stable_params,
                'overall_sensitivity': 'High' if len(sensitive_params) > 2 else 'Medium' if sensitive_params else 'Low',
                'tuning_priority': sensitive_params[:2] if sensitive_params else ['max_features', 'n_estimators'],
                'extra_trees_notes': extra_trees_notes
            }
            
        except:
            return {'overall_sensitivity': 'Unknown'}
    
    def _analyze_computational_efficiency(self):
        """Analyze computational efficiency of Extra Trees"""
        if not self.computational_efficiency_analysis:
            return
        
        try:
            import time
            
            # Measure training time components
            start_time = time.time()
            
            # Create small model for timing analysis
            timing_model = ExtraTreesRegressor(
                n_estimators=20,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=1
            )
            
            # Sample data for timing
            sample_size = min(500, len(self.X_original_))
            indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
            X_sample = self.X_original_[indices]
            y_sample = self.y_original_[indices]
            
            # Measure training time
            train_start = time.time()
            timing_model.fit(X_sample, y_sample)
            training_time = time.time() - train_start
            
            # Measure prediction time
            pred_start = time.time()
            predictions = timing_model.predict(X_sample)
            prediction_time = time.time() - pred_start
            
            # Calculate efficiency metrics
            samples_per_second_train = sample_size / training_time
            samples_per_second_predict = sample_size / prediction_time
            
            # Compare with Random Forest (if comparison enabled)
            rf_comparison = None
            if self.compare_with_random_forest:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    
                    rf_model = RandomForestRegressor(
                        n_estimators=20,
                        max_depth=5,
                        random_state=self.random_state,
                        n_jobs=1
                    )
                    
                    rf_train_start = time.time()
                    rf_model.fit(X_sample, y_sample)
                    rf_training_time = time.time() - rf_train_start
                    
                    rf_pred_start = time.time()
                    rf_predictions = rf_model.predict(X_sample)
                    rf_prediction_time = time.time() - rf_pred_start
                    
                    rf_comparison = {
                        'rf_training_time': rf_training_time,
                        'rf_prediction_time': rf_prediction_time,
                        'speedup_training': rf_training_time / training_time,
                        'speedup_prediction': rf_prediction_time / prediction_time
                    }
                    
                except:
                    rf_comparison = {'error': 'Could not compare with Random Forest'}
            
            # Efficiency assessment
            efficiency_assessment = self._assess_computational_efficiency(
                training_time, prediction_time, samples_per_second_train, rf_comparison
            )
            
            self.computational_efficiency_analysis_ = {
                'training_time': training_time,
                'prediction_time': prediction_time,
                'samples_per_second_train': samples_per_second_train,
                'samples_per_second_predict': samples_per_second_predict,
                'sample_size_tested': sample_size,
                'rf_comparison': rf_comparison,
                'efficiency_assessment': efficiency_assessment
            }
            
        except Exception as e:
            self.computational_efficiency_analysis_ = {
                'error': f'Could not analyze computational efficiency: {str(e)}'
            }
    
    def _assess_computational_efficiency(self, training_time, prediction_time, samples_per_second, rf_comparison):
        """Assess computational efficiency of Extra Trees"""
        try:
            # Base efficiency assessment
            if samples_per_second > 1000:
                efficiency_level = "Excellent - Very fast training"
            elif samples_per_second > 500:
                efficiency_level = "Good - Fast training"
            elif samples_per_second > 100:
                efficiency_level = "Moderate - Acceptable training speed"
            else:
                efficiency_level = "Slow - Consider reducing complexity"
            
            advantages = [
                "No best split search - uses random thresholds",
                "Faster tree construction than Random Forest",
                "Parallel training across trees",
                "Simpler split selection process"
            ]
            
            # Random Forest comparison
            rf_advantage = None
            if rf_comparison and 'speedup_training' in rf_comparison:
                speedup = rf_comparison['speedup_training']
                if speedup > 1.2:
                    rf_advantage = f"Extra Trees is {speedup:.1f}x faster than Random Forest"
                elif speedup > 0.8:
                    rf_advantage = "Extra Trees and Random Forest have similar speed"
                else:
                    rf_advantage = f"Random Forest is {1/speedup:.1f}x faster (unexpected)"
            
            return {
                'efficiency_level': efficiency_level,
                'training_speed': f"{samples_per_second:.0f} samples/second",
                'advantages': advantages,
                'rf_comparison': rf_advantage,
                'recommendations': self._get_efficiency_recommendations(samples_per_second)
            }
            
        except:
            return {'efficiency_level': 'Unknown'}
    
    def _get_efficiency_recommendations(self, samples_per_second):
        """Get recommendations for improving computational efficiency"""
        try:
            recommendations = []
            
            if samples_per_second < 100:
                recommendations.extend([
                    "Reduce n_estimators for faster training",
                    "Set max_depth to limit tree size",
                    "Increase min_samples_split to reduce tree complexity",
                    "Use n_jobs=-1 for parallel processing"
                ])
            elif samples_per_second < 500:
                recommendations.extend([
                    "Consider increasing n_jobs for better parallelization",
                    "Monitor tree depth and complexity"
                ])
            else:
                recommendations.append("Efficiency is good - current settings are optimal")
            
            return recommendations
            
        except:
            return ["Could not generate efficiency recommendations"]
    
    def _compare_with_random_forest(self):
        """Compare Extra Trees with Random Forest performance"""
        if not self.compare_with_random_forest:
            return
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Create comparable Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=min(50, self.n_estimators),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=1,
                bootstrap=True  # RF typically uses bootstrap
            )
            
            # Cross-validation comparison
            et_scores = cross_val_score(
                self.model_, self.X_original_, self.y_original_,
                cv=3, scoring='r2'
            )
            
            rf_scores = cross_val_score(
                rf_model, self.X_original_, self.y_original_,
                cv=3, scoring='r2'
            )
            
            # Performance comparison
            et_mean = np.mean(et_scores)
            rf_mean = np.mean(rf_scores)
            performance_difference = et_mean - rf_mean
            
            # Statistical significance test
            from scipy import stats
            try:
                t_stat, p_value = stats.ttest_rel(et_scores, rf_scores)
                significant_difference = p_value < 0.05
            except:
                significant_difference = None
                p_value = None
            
            # Comparison assessment
            comparison_assessment = self._assess_rf_comparison(
                et_mean, rf_mean, performance_difference, significant_difference
            )
            
            self.random_forest_comparison_ = {
                'extra_trees_scores': et_scores,
                'random_forest_scores': rf_scores,
                'extra_trees_mean': et_mean,
                'random_forest_mean': rf_mean,
                'performance_difference': performance_difference,
                'significant_difference': significant_difference,
                'p_value': p_value,
                'comparison_assessment': comparison_assessment
            }
            
        except Exception as e:
            self.random_forest_comparison_ = {
                'error': f'Could not compare with Random Forest: {str(e)}'
            }
    
    def _assess_rf_comparison(self, et_score, rf_score, difference, significant):
        """Assess Extra Trees vs Random Forest comparison"""
        try:
            if difference > 0.05:
                if significant:
                    performance_assessment = "Extra Trees significantly outperforms Random Forest"
                else:
                    performance_assessment = "Extra Trees appears better but not statistically significant"
            elif difference < -0.05:
                if significant:
                    performance_assessment = "Random Forest significantly outperforms Extra Trees"
                else:
                    performance_assessment = "Random Forest appears better but not statistically significant"
            else:
                performance_assessment = "Extra Trees and Random Forest have similar performance"
            
            # When to prefer Extra Trees
            et_advantages = [
                "Faster training due to random splits",
                "Better for high-dimensional data",
                "Less prone to overfitting with extreme randomness",
                "Good when interpretability is less critical"
            ]
            
            # When to prefer Random Forest
            rf_advantages = [
                "More interpretable individual trees",
                "Better for smaller datasets",
                "More established and well-understood",
                "Built-in out-of-bag evaluation"
            ]
            
            return {
                'performance_assessment': performance_assessment,
                'extra_trees_advantages': et_advantages,
                'random_forest_advantages': rf_advantages,
                'recommendation': self._get_algorithm_recommendation(difference, significant)
            }
            
        except:
            return {'performance_assessment': 'Unknown'}
    
    def _get_algorithm_recommendation(self, difference, significant):
        """Get recommendation on which algorithm to use"""
        try:
            if difference > 0.03 and significant:
                return "Prefer Extra Trees for this dataset - significant performance advantage"
            elif difference < -0.03 and significant:
                return "Prefer Random Forest for this dataset - significant performance advantage"
            elif difference > 0.01:
                return "Slight advantage to Extra Trees - consider speed vs interpretability tradeoffs"
            elif difference < -0.01:
                return "Slight advantage to Random Forest - consider stability vs speed tradeoffs"
            else:
                return "Similar performance - choose based on computational requirements and interpretability needs"
        except:
            return "Could not generate algorithm recommendation"
    
    def _analyze_randomness_impact(self):
        """Analyze the impact of extreme randomness on performance"""
        if not self.randomness_impact_analysis:
            return
        
        try:
            # Test different levels of randomness by varying max_features
            randomness_levels = ['sqrt', 'log2', None, 0.3, 0.7]
            randomness_results = []
            
            for max_feat in randomness_levels[:3]:  # Limit for efficiency
                try:
                    model = ExtraTreesRegressor(
                        n_estimators=30,
                        max_features=max_feat,
                        random_state=self.random_state,
                        n_jobs=1
                    )
                    
                    scores = cross_val_score(
                        model, self.X_original_, self.y_original_,
                        cv=3, scoring='r2'
                    )
                    
                    randomness_results.append({
                        'max_features': max_feat,
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'randomness_level': self._assess_randomness_level(max_feat)
                    })
                    
                except:
                    continue
            
            # Analyze randomness impact
            randomness_analysis = self._analyze_randomness_effects(randomness_results)
            
            self.randomness_impact_analysis_ = {
                'randomness_results': randomness_results,
                'randomness_analysis': randomness_analysis
            }
            
        except Exception as e:
            self.randomness_impact_analysis_ = {
                'error': f'Could not analyze randomness impact: {str(e)}'
            }
    
    def _assess_randomness_level(self, max_features):
        """Assess the level of randomness for given max_features"""
        if max_features is None:
            return "Low"
        elif max_features == 'sqrt':
            return "Medium"
        elif max_features == 'log2':
            return "High"
        elif isinstance(max_features, float) and max_features < 0.5:
            return "Very High"
        else:
            return "Medium"
    
    def _analyze_randomness_effects(self, results):
        """Analyze the effects of different randomness levels"""
        try:
            if len(results) < 2:
                return {'status': 'Insufficient data for randomness analysis'}
            
            # Find optimal randomness level
            best_result = max(results, key=lambda x: x['mean_score'])
            
            # Analyze variance vs performance tradeoff
            performances = [r['mean_score'] for r in results]
            variances = [r['std_score'] for r in results]
            
            # Correlation between randomness and performance
            randomness_scores = [3 if r['randomness_level'] == 'Very High' else 
                               2 if r['randomness_level'] == 'High' else
                               1 if r['randomness_level'] == 'Medium' else 0
                               for r in results]
            
            try:
                correlation = np.corrcoef(randomness_scores, performances)[0, 1]
            except:
                correlation = 0.0
            
            return {
                'optimal_randomness': best_result['max_features'],
                'optimal_randomness_level': best_result['randomness_level'],
                'best_performance': best_result['mean_score'],
                'randomness_performance_correlation': correlation,
                'interpretation': self._interpret_randomness_correlation(correlation)
            }
            
        except:
            return {'status': 'Error in randomness analysis'}
    
    def _interpret_randomness_correlation(self, correlation):
        """Interpret the correlation between randomness and performance"""
        if correlation > 0.5:
            return "Higher randomness improves performance - dataset benefits from extreme randomization"
        elif correlation > 0.2:
            return "Moderate positive effect of randomness - Extra Trees advantage confirmed"
        elif correlation > -0.2:
            return "Randomness has minimal impact - performance stable across settings"
        else:
            return "Higher randomness reduces performance - consider Random Forest instead"
    
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
            
            # Extra Trees performance
            if 'r2' in self.cross_validation_analysis_.get('cv_results', {}):
                et_performance = self.cross_validation_analysis_['cv_results']['r2']['mean']
            else:
                et_performance = self.model_.score(self.X_original_, self.y_original_)
            
            # Performance improvements
            improvements = {}
            for name, result in benchmark_results.items():
                if 'cv_r2_mean' in result:
                    improvement = et_performance - result['cv_r2_mean']
                    improvements[name] = {
                        'absolute_improvement': improvement,
                        'relative_improvement': improvement / max(0.01, abs(result['cv_r2_mean']))
                    }
            
            self.performance_benchmarking_ = {
                'extra_trees_r2': et_performance,
                'benchmark_results': benchmark_results,
                'improvements': improvements,
                'performance_ranking': self._rank_performance(et_performance, benchmark_results)
            }
            
        except Exception as e:
            self.performance_benchmarking_ = {
                'error': f'Could not analyze performance benchmarks: {str(e)}'
            }
    
    def _rank_performance(self, et_score, benchmark_results):
        """Rank Extra Trees performance against benchmarks"""
        try:
            all_scores = [et_score]
            model_names = ['Extra Trees']
            
            for name, result in benchmark_results.items():
                if 'cv_r2_mean' in result:
                    all_scores.append(result['cv_r2_mean'])
                    model_names.append(name.replace('_', ' ').title())
            
            # Sort by score
            sorted_indices = np.argsort(all_scores)[::-1]
            ranking = [(model_names[i], all_scores[i]) for i in sorted_indices]
            
            et_rank = next(i for i, (name, _) in enumerate(ranking) if name == 'Extra Trees') + 1
            
            return {
                'ranking': ranking,
                'extra_trees_rank': et_rank,
                'total_models': len(ranking),
                'performance_percentile': (len(ranking) - et_rank + 1) / len(ranking) * 100
            }
            
        except:
            return {'error': 'Could not rank performance'}
    
    def _analyze_split_quality(self):
        """Analyze quality of random splits"""
        if not self.split_quality_analysis:
            return
        
        try:
            # Sample trees for analysis
            sample_trees = self.model_.estimators_[:5]
            
            split_qualities = []
            
            for tree in sample_trees:
                tree_structure = tree.tree_
                
                # Extract split information
                features = tree_structure.feature[tree_structure.feature >= 0]
                thresholds = tree_structure.threshold[tree_structure.feature >= 0]
                impurities = tree_structure.impurity[tree_structure.feature >= 0]
                
                if len(features) > 0:
                    # Calculate split quality metrics
                    impurity_decreases = []
                    for i in range(len(features)):
                        left_child = tree_structure.children_left[i]
                        right_child = tree_structure.children_right[i]
                        
                        if left_child != -1 and right_child != -1:
                            left_impurity = tree_structure.impurity[left_child]
                            right_impurity = tree_structure.impurity[right_child]
                            left_samples = tree_structure.n_node_samples[left_child]
                            right_samples = tree_structure.n_node_samples[right_child]
                            total_samples = left_samples + right_samples
                            
                            weighted_child_impurity = (left_samples * left_impurity + 
                                                     right_samples * right_impurity) / total_samples
                            impurity_decrease = impurities[i] - weighted_child_impurity
                            impurity_decreases.append(impurity_decrease)
                    
                    split_qualities.append({
                        'mean_impurity_decrease': np.mean(impurity_decreases),
                        'std_impurity_decrease': np.std(impurity_decreases),
                        'n_splits': len(impurity_decreases)
                    })
            
            # Aggregate split quality metrics
            if split_qualities:
                split_quality_summary = {
                    'mean_impurity_decrease': np.mean([sq['mean_impurity_decrease'] for sq in split_qualities]),
                    'overall_std': np.mean([sq['std_impurity_decrease'] for sq in split_qualities]),
                    'consistency_score': 1.0 / (1.0 + np.std([sq['mean_impurity_decrease'] for sq in split_qualities])),
                    'trees_analyzed': len(split_qualities)
                }
                
                # Quality assessment
                quality_assessment = self._assess_split_quality(split_quality_summary)
                
                self.split_quality_analysis_ = {
                    'individual_trees': split_qualities,
                    'summary': split_quality_summary,
                    'quality_assessment': quality_assessment
                }
            else:
                self.split_quality_analysis_ = {
                    'error': 'No valid splits found for analysis'
                }
            
        except Exception as e:
            self.split_quality_analysis_ = {
                'error': f'Could not analyze split quality: {str(e)}'
            }
    
    def _assess_split_quality(self, summary):
        """Assess the quality of random splits"""
        try:
            consistency = summary['consistency_score']
            mean_decrease = summary['mean_impurity_decrease']
            
            if consistency > 0.8 and mean_decrease > 0.01:
                quality_level = "High - Random splits are effective"
            elif consistency > 0.6 and mean_decrease > 0.005:
                quality_level = "Good - Acceptable split quality"
            elif consistency > 0.4:
                quality_level = "Moderate - Random splits show some benefit"
            else:
                quality_level = "Low - Random splits may not be optimal for this data"
            
            return {
                'quality_level': quality_level,
                'consistency_score': consistency,
                'mean_impurity_decrease': mean_decrease,
                'interpretation': 'Higher consistency indicates stable random split benefits'
            }
            
        except:
            return {'quality_level': 'Unknown'}
    
    def _analyze_randomness_efficiency(self):
        """Analyze efficiency of randomness in Extra Trees"""
        if not self.randomness_efficiency_analysis:
            return
        
        try:
            # Compare random vs best split efficiency (conceptual analysis)
            # This is estimated since we can't directly compare with best splits
            
            # Analyze tree complexity resulting from random splits
            tree_complexities = []
            
            for tree in self.model_.estimators_[:10]:
                tree_structure = tree.tree_
                
                complexity_metrics = {
                    'depth': tree_structure.max_depth,
                    'nodes': tree_structure.node_count,
                    'leaves': tree_structure.n_leaves,
                    'depth_to_leaves_ratio': tree_structure.max_depth / tree_structure.n_leaves if tree_structure.n_leaves > 0 else 0
                }
                
                tree_complexities.append(complexity_metrics)
            
            # Aggregate complexity analysis
            complexity_summary = {
                'mean_depth': np.mean([tc['depth'] for tc in tree_complexities]),
                'mean_nodes': np.mean([tc['nodes'] for tc in tree_complexities]),
                'mean_leaves': np.mean([tc['leaves'] for tc in tree_complexities]),
                'complexity_variance': np.var([tc['nodes'] for tc in tree_complexities]),
                'efficiency_score': np.mean([tc['depth_to_leaves_ratio'] for tc in tree_complexities])
            }
            
            # Efficiency assessment
            efficiency_assessment = self._assess_randomness_efficiency(complexity_summary)
            
            self.randomness_efficiency_analysis_ = {
                'tree_complexities': tree_complexities,
                'complexity_summary': complexity_summary,
                'efficiency_assessment': efficiency_assessment
            }
            
        except Exception as e:
            self.randomness_efficiency_analysis_ = {
                'error': f'Could not analyze randomness efficiency: {str(e)}'
            }
    
    def _assess_randomness_efficiency(self, summary):
        """Assess efficiency of random splits"""
        try:
            efficiency_score = summary['efficiency_score']
            complexity_variance = summary['complexity_variance']
            
            if efficiency_score > 0.8 and complexity_variance > 100:
                efficiency_level = "High - Random splits create diverse, efficient trees"
            elif efficiency_score > 0.6:
                efficiency_level = "Good - Randomness provides reasonable efficiency"
            elif efficiency_score > 0.4:
                efficiency_level = "Moderate - Some efficiency from random splits"
            else:
                efficiency_level = "Low - Random splits may be creating inefficient trees"
            
            return {
                'efficiency_level': efficiency_level,
                'efficiency_score': efficiency_score,
                'complexity_variance': complexity_variance,
                'interpretation': 'Higher scores indicate better tree structure from random splits'
            }
            
        except:
            return {'efficiency_level': 'Unknown'}
    
    def _analyze_bias_variance(self):
        """Analyze bias-variance tradeoff in Extra Trees"""
        if not self.bias_variance_analysis:
            return
        
        try:
            # Simplified bias-variance analysis using bootstrap samples
            n_bootstrap = 20
            sample_size = min(200, len(self.X_original_))
            
            # Create test data for analysis
            test_indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
            X_test = self.X_original_[test_indices]
            y_test = self.y_original_[test_indices]
            
            # Train multiple models on bootstrap samples
            predictions = []
            
            for i in range(n_bootstrap):
                # Bootstrap sample
                boot_indices = np.random.choice(len(self.X_original_), len(self.X_original_), replace=True)
                X_boot = self.X_original_[boot_indices]
                y_boot = self.y_original_[boot_indices]
                
                # Train model
                model = ExtraTreesRegressor(
                    n_estimators=30,
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    random_state=i,
                    n_jobs=1
                )
                
                model.fit(X_boot, y_boot)
                pred = model.predict(X_test)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate bias and variance
            mean_prediction = np.mean(predictions, axis=0)
            variance = np.mean(np.var(predictions, axis=0))
            bias_squared = np.mean((mean_prediction - y_test) ** 2)
            
            # Total error decomposition
            total_error = bias_squared + variance
            noise_estimate = np.var(y_test) * 0.1  # Rough noise estimate
            
            bias_variance_analysis = {
                'bias_squared': bias_squared,
                'variance': variance,
                'total_error': total_error,
                'noise_estimate': noise_estimate,
                'bias_variance_ratio': bias_squared / (variance + 1e-10),
                'variance_fraction': variance / total_error,
                'bias_fraction': bias_squared / total_error
            }
            
            # Assessment
            bv_assessment = self._assess_bias_variance(bias_variance_analysis)
            
            self.bias_variance_analysis_ = {
                'bias_variance_decomposition': bias_variance_analysis,
                'assessment': bv_assessment,
                'n_bootstrap_samples': n_bootstrap,
                'test_sample_size': sample_size
            }
            
        except Exception as e:
            self.bias_variance_analysis_ = {
                'error': f'Could not analyze bias-variance tradeoff: {str(e)}'
            }
    
    def _assess_bias_variance(self, analysis):
        """Assess bias-variance tradeoff"""
        try:
            bias_fraction = analysis['bias_fraction']
            variance_fraction = analysis['variance_fraction']
            ratio = analysis['bias_variance_ratio']
            
            if variance_fraction > 0.7:
                assessment = "High variance - model might benefit from more regularization"
                recommendation = "Consider reducing max_features or increasing min_samples_split"
            elif bias_fraction > 0.7:
                assessment = "High bias - model might be too simple"
                recommendation = "Consider increasing model complexity or reducing min_samples_leaf"
            elif 0.3 <= variance_fraction <= 0.7:
                assessment = "Good bias-variance balance"
                recommendation = "Current settings provide good tradeoff"
            else:
                assessment = "Moderate bias-variance tradeoff"
                recommendation = "Fine-tune regularization parameters"
            
            return {
                'assessment': assessment,
                'recommendation': recommendation,
                'bias_dominance': bias_fraction > variance_fraction,
                'variance_dominance': variance_fraction > bias_fraction,
                'balance_score': 1.0 - abs(bias_fraction - variance_fraction)
            }
            
        except:
            return {'assessment': 'Unknown bias-variance characteristics'}
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Ensemble Configuration", "Tree Parameters", "Extra Trees Settings", "Analysis Options", "Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Extra Trees Ensemble Configuration**")
            
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
                    help="Whether bootstrap samples are used (default False for Extra Trees)",
                    key=f"{key_prefix}_bootstrap"
                )
                
                n_jobs = st.selectbox(
                    "Parallel Jobs:",
                    options=[-1, 1, 2, 4],
                    index=[-1, 1, 2, 4].index(self.n_jobs) if self.n_jobs in [-1, 1, 2, 4] else 0,
                    help="Number of jobs for parallel processing (-1 = all cores)",
                    key=f"{key_prefix}_n_jobs"
                )
            
            with col2:
                max_samples = st.selectbox(
                    "Max Samples per Tree:",
                    options=[None, 0.7, 0.8, 0.9, 1.0],
                    index=0 if self.max_samples is None else [None, 0.7, 0.8, 0.9, 1.0].index(self.max_samples),
                    help="Maximum number of samples to draw for each tree",
                    key=f"{key_prefix}_max_samples"
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
                
                verbose = st.selectbox(
                    "Verbose Output:",
                    options=[0, 1, 2],
                    index=self.verbose,
                    help="Control the verbosity of training output",
                    key=f"{key_prefix}_verbose"
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
            st.markdown("**Extra Trees Specific Settings**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_features = st.selectbox(
                    "Max Features per Split:",
                    options=['sqrt', 'log2', None, 0.3, 0.5, 0.7],
                    index=['sqrt', 'log2', None, 0.3, 0.5, 0.7].index(self.max_features) if self.max_features in ['sqrt', 'log2', None, 0.3, 0.5, 0.7] else 0,
                    help="Number of features for random selection at each split",
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
                
                compare_with_random_forest = st.checkbox(
                    "Compare with Random Forest",
                    value=self.compare_with_random_forest,
                    help="Perform comparison analysis with Random Forest",
                    key=f"{key_prefix}_compare_with_random_forest"
                )
            
            with col2:
                st.markdown("**Extra Trees Advantages:**")
                st.info("""
                🎲 **Random Thresholds**: Uses random splits instead of optimal splits
                ⚡ **Faster Training**: No best split search required
                🎯 **Better Generalization**: Extreme randomness reduces overfitting
                📊 **High-Dimensional Data**: Excels with many features
                🔄 **Lower Variance**: More stable predictions than Random Forest
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
                
                split_randomness_analysis = st.checkbox(
                    "Split Randomness Analysis",
                    value=self.split_randomness_analysis,
                    help="Analyze impact of random splits",
                    key=f"{key_prefix}_split_randomness_analysis"
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
                
                randomness_impact_analysis = st.checkbox(
                    "Randomness Impact Analysis",
                    value=self.randomness_impact_analysis,
                    help="Analyze impact of different randomness levels",
                    key=f"{key_prefix}_randomness_impact_analysis"
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
                
                split_quality_analysis = st.checkbox(
                    "Split Quality Analysis",
                    value=self.split_quality_analysis,
                    help="Analyze quality of random splits",
                    key=f"{key_prefix}_split_quality_analysis"
                )
                
                bias_variance_analysis = st.checkbox(
                    "Bias-Variance Analysis",
                    value=self.bias_variance_analysis,
                    help="Analyze bias-variance tradeoff",
                    key=f"{key_prefix}_bias_variance_analysis"
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
            st.markdown("**Extra Trees Regressor - Extremely Randomized Trees**")
            
            # Algorithm information
            if st.button("📚 Algorithm Information", key=f"{key_prefix}_algo_info"):
                st.markdown("""
                **Extra Trees (Extremely Randomized Trees) - Advanced Ensemble Method**
                
                Extra Trees extends Random Forest by introducing extreme randomness in the tree 
                building process, using random thresholds for splits rather than searching for 
                optimal ones. This leads to faster training and often better generalization.
                
                **Core Principles:**
                • **Random Feature Selection** - Random subset of features at each split (like RF)
                • **Random Threshold Selection** - Random thresholds instead of optimal splits
                • **No Bootstrap** - Uses entire dataset by default (unlike Random Forest)
                • **Extreme Randomization** - Maximum randomness in tree construction
                • **Ensemble Averaging** - Final prediction is average of all tree predictions
                • **Faster Training** - No expensive best-split search required
                
                **Key Advantages:**
                • ⚡ **Faster Training** - Random splits eliminate best-split search
                • 🎯 **Better Generalization** - Extreme randomness reduces overfitting
                • 📊 **High-Dimensional Data** - Excels with many features
                • 🔄 **Lower Variance** - More stable than Random Forest
                • 🛡️ **Robust to Noise** - Random splits are less sensitive to outliers
                • 📈 **No Assumptions** - Handles non-linear relationships naturally
                """)
            
            # When to use Extra Trees
            if st.button("🎯 When to Use Extra Trees", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • High-dimensional datasets with many features
                • Need for fast training with good performance
                • Non-linear relationships in data
                • Overfitting issues with other tree methods
                • Computational efficiency is important
                
                **Data Characteristics:**
                • Large feature spaces (hundreds to thousands of features)
                • Moderate to large sample sizes
                • Noisy data or outliers present
                • Complex feature interactions
                • Time-sensitive training requirements
                
                **Examples:**
                • Genomics and bioinformatics (high-dimensional data)
                • Text mining and NLP feature extraction
                • Computer vision feature analysis
                • High-frequency financial modeling
                • Large-scale scientific simulations
                • Real-time recommendation systems
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Extremely fast training (faster than Random Forest)
                ✅ Often better generalization than Random Forest
                ✅ Excellent for high-dimensional data
                ✅ Built-in feature importance ranking
                ✅ Robust to overfitting due to extreme randomness
                ✅ Handles large feature spaces efficiently
                ✅ Lower variance than Random Forest
                ✅ No hyperparameter tuning for split selection
                ✅ Parallelizable training and prediction
                ✅ Good performance with default parameters
                
                **Limitations:**
                ❌ Individual trees are less interpretable (random splits)
                ❌ May underperform on small, simple datasets
                ❌ Random splits can miss obvious optimal splits
                ❌ Less control over individual tree quality
                ❌ May require more trees than Random Forest
                ❌ Can struggle with very structured data
                ❌ Less established than Random Forest
                """)
            
            # Extra Trees vs other methods
            if st.button("🔍 Extra Trees vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **Extra Trees vs Other Ensemble Methods:**
                
                **Extra Trees vs Random Forest:**
                • ET: Faster training, random thresholds, no bootstrap by default
                • RF: Optimal splits, bootstrap sampling, more interpretable trees
                • ET: Better for high-dimensional data, often superior generalization
                • RF: Better for smaller datasets, more predictable behavior
                
                **Extra Trees vs Gradient Boosting:**
                • ET: Parallel training, less prone to overfitting
                • GB: Sequential training, often higher accuracy but slower
                • ET: Better bias-variance tradeoff, more robust
                • GB: Better for structured data, requires careful tuning
                
                **Extra Trees vs Single Decision Tree:**
                • ET: Much better generalization, ensemble robustness
                • Tree: More interpretable, faster prediction per sample
                • ET: Handles complex patterns, feature importance
                • Tree: Simple rules, easier to explain decisions
                
                **Extra Trees vs Linear Models:**
                • ET: Handles non-linearity and interactions automatically
                • Linear: More interpretable, better extrapolation
                • ET: No assumptions about relationships
                • Linear: Faster training on large datasets
                
                **Extra Trees vs Neural Networks:**
                • ET: Better for tabular data, automatic feature selection
                • NN: Better for images, sequences, complex patterns
                • ET: Requires less data, easier to tune
                • NN: More flexible, can model any function
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Extra Trees Best Practices:**
                
                **Ensemble Configuration:**
                1. **n_estimators**: Start with 100, increase to 200-500 for better performance
                2. **max_features**: Use 'sqrt' for balanced randomness, 'log2' for more diversity
                3. **bootstrap=False**: Default setting leverages full dataset
                4. **max_samples**: Only relevant if bootstrap=True
                
                **Tree Parameters:**
                1. **max_depth**: Usually leave as None, use min_samples_split for control
                2. **min_samples_split**: 2-10 depending on dataset size
                3. **min_samples_leaf**: 1-3, higher values for more regularization
                4. **min_impurity_decrease**: Use sparingly, random splits are already regularized
                
                **Performance Optimization:**
                1. Use **n_jobs=-1** for parallel training
                2. Monitor convergence to determine optimal n_estimators
                3. Use feature importance for feature selection
                4. Compare with Random Forest to validate choice
                
                **Model Validation:**
                1. Use cross-validation for robust performance assessment
                2. Analyze randomness impact on your specific dataset
                3. Check tree diversity for ensemble quality
                4. Monitor computational efficiency vs accuracy tradeoff
                
                **When to Tune:**
                1. **max_features**: Critical parameter affecting randomness level
                2. **n_estimators**: More trees usually help until convergence
                3. **min_samples_split**: Controls tree complexity and speed
                4. **max_depth**: Only if default unlimited depth causes issues
                """)
            
            # Advanced techniques
            if st.button("🚀 Advanced Techniques", key=f"{key_prefix}_advanced"):
                st.markdown("""
                **Advanced Extra Trees Techniques:**
                
                **Randomness Optimization:**
                • **Adaptive Feature Selection**: Dynamically adjust max_features based on data
                • **Threshold Distribution Analysis**: Study optimal threshold distributions
                • **Multi-level Randomness**: Combine different randomness strategies
                • **Random Split Quality Assessment**: Evaluate effectiveness of random splits
                
                **Ensemble Enhancement:**
                • **Weighted Ensemble**: Weight trees based on individual performance
                • **Dynamic Ensemble Sizing**: Adaptive number of estimators
                • **Hybrid Ensembles**: Combine Extra Trees with other methods
                • **Selective Ensemble**: Choose best-performing trees post-training
                
                **Computational Optimization:**
                • **Progressive Training**: Add trees incrementally with early stopping
                • **Parallel Feature Engineering**: Optimize feature computation
                • **Memory Optimization**: Reduce memory footprint for large ensembles
                • **Incremental Learning**: Update ensemble with new data
                
                **Specialized Applications:**
                • **Time Series Ensembles**: Adapt for temporal data
                • **Multi-target Regression**: Handle multiple outputs simultaneously
                • **Online Learning**: Continuous model updates
                • **Quantile Regression**: Predict confidence intervals
                
                **Interpretation and Analysis:**
                • **Feature Interaction Discovery**: Use ensemble for feature engineering
                • **Uncertainty Quantification**: Leverage tree disagreement
                • **Model Distillation**: Create simpler models from ensemble
                • **Decision Path Analysis**: Understand ensemble decision patterns
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
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "compute_feature_importance": compute_feature_importance,
            "compute_permutation_importance": compute_permutation_importance,
            "split_randomness_analysis": split_randomness_analysis,
            "tree_diversity_analysis": tree_diversity_analysis,
            "ensemble_convergence_analysis": ensemble_convergence_analysis,
            "feature_interaction_analysis": feature_interaction_analysis,
            "learning_curve_analysis": learning_curve_analysis,
            "hyperparameter_sensitivity_analysis": hyperparameter_sensitivity_analysis,
            "computational_efficiency_analysis": computational_efficiency_analysis,
            "compare_with_random_forest": compare_with_random_forest,
            "randomness_impact_analysis": randomness_impact_analysis,
            "visualize_trees": visualize_trees,
            "max_trees_to_visualize": max_trees_to_visualize,
            "feature_importance_analysis": True,
            "prediction_distribution_analysis": True,
            "cross_validation_analysis": cross_validation_analysis,
            "cv_folds": cv_folds,
            "performance_benchmarking": performance_benchmarking,
            "split_quality_analysis": split_quality_analysis,
            "randomness_efficiency_analysis": True,
            "bias_variance_analysis": bias_variance_analysis
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return ExtraTreesRegressorPlugin(
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
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            random_state=hyperparameters.get("random_state", self.random_state),
            verbose=hyperparameters.get("verbose", self.verbose),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            ccp_alpha=hyperparameters.get("ccp_alpha", self.ccp_alpha),
            max_samples=hyperparameters.get("max_samples", self.max_samples),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            compute_permutation_importance=hyperparameters.get("compute_permutation_importance", self.compute_permutation_importance),
            split_randomness_analysis=hyperparameters.get("split_randomness_analysis", self.split_randomness_analysis),
            tree_diversity_analysis=hyperparameters.get("tree_diversity_analysis", self.tree_diversity_analysis),
            ensemble_convergence_analysis=hyperparameters.get("ensemble_convergence_analysis", self.ensemble_convergence_analysis),
            feature_interaction_analysis=hyperparameters.get("feature_interaction_analysis", self.feature_interaction_analysis),
            learning_curve_analysis=hyperparameters.get("learning_curve_analysis", self.learning_curve_analysis),
            hyperparameter_sensitivity_analysis=hyperparameters.get("hyperparameter_sensitivity_analysis", self.hyperparameter_sensitivity_analysis),
            computational_efficiency_analysis=hyperparameters.get("computational_efficiency_analysis", self.computational_efficiency_analysis),
            compare_with_random_forest=hyperparameters.get("compare_with_random_forest", self.compare_with_random_forest),
            randomness_impact_analysis=hyperparameters.get("randomness_impact_analysis", self.randomness_impact_analysis),
            visualize_trees=hyperparameters.get("visualize_trees", self.visualize_trees),
            max_trees_to_visualize=hyperparameters.get("max_trees_to_visualize", self.max_trees_to_visualize),
            feature_importance_analysis=hyperparameters.get("feature_importance_analysis", self.feature_importance_analysis),
            prediction_distribution_analysis=hyperparameters.get("prediction_distribution_analysis", self.prediction_distribution_analysis),
            cross_validation_analysis=hyperparameters.get("cross_validation_analysis", self.cross_validation_analysis),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            performance_benchmarking=hyperparameters.get("performance_benchmarking", self.performance_benchmarking),
            split_quality_analysis=hyperparameters.get("split_quality_analysis", self.split_quality_analysis),
            randomness_efficiency_analysis=hyperparameters.get("randomness_efficiency_analysis", self.randomness_efficiency_analysis),
            bias_variance_analysis=hyperparameters.get("bias_variance_analysis", self.bias_variance_analysis)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Extra Trees (minimal preprocessing needed)"""
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
        """Check if Extra Trees is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Extra Trees requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Extra Trees Regressor requires continuous numerical target values"
            
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
                advantages.append(f"Good sample size ({n_samples}) - adequate for Extra Trees ensemble")
            else:
                considerations.append(f"Small sample size ({n_samples}) - ensemble may have limited diversity")
            
            # Feature dimensionality assessment (Extra Trees excel with high dimensions)
            if n_features >= 100:
                advantages.append(f"High dimensionality ({n_features}) - ideal for Extra Trees' random feature selection")
            elif n_features >= 20:
                advantages.append(f"Moderate dimensionality ({n_features}) - good for Extra Trees randomness")
            else:
                considerations.append(f"Low dimensionality ({n_features}) - Extra Trees advantage may be limited")
            
            # Data characteristics favorable to Extra Trees
            advantages.append("Extremely randomized trees with superior generalization")
            advantages.append("Faster training than Random Forest due to random splits")
            advantages.append("Excellent for high-dimensional data and feature selection")
            advantages.append("Robust to overfitting through extreme randomness")
            
            # Check feature-to-sample ratio (Extra Trees handle high-dimensional data well)
            feature_sample_ratio = n_features / n_samples
            if feature_sample_ratio > 0.5:
                advantages.append(f"High feature-to-sample ratio ({feature_sample_ratio:.2f}) - Extra Trees excel in this scenario")
            elif feature_sample_ratio > 0.1:
                advantages.append(f"Moderate feature-to-sample ratio ({feature_sample_ratio:.2f}) - good for Extra Trees")
            else:
                considerations.append(f"Low feature-to-sample ratio ({feature_sample_ratio:.2f}) - consider simpler models")
            
            # Computational efficiency advantages
            if n_samples > 10000 or n_features > 100:
                advantages.append("Large dataset - benefits from Extra Trees' computational efficiency")
            
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
                f"🌳 Suitability for Extra Trees: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance with Extra Trees specific insights"""
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
                    'randomness_robustness': analysis['importance_stability']['randomness_robustness'] if 'importance_stability' in analysis else 0.0
                }
            
            return {
                'feature_importance': feature_importance,
                'top_features': analysis['top_features'],
                'importance_analysis': analysis,
                'ensemble_size': len(self.model_.estimators_)
            }
        
    def _get_efficiency_recommendations(self, samples_per_second):
        """Get recommendations for improving computational efficiency"""
        try:
            recommendations = []
            
            if samples_per_second < 100:
                recommendations.extend([
                    "Reduce n_estimators for faster training",
                    "Set max_depth to limit tree size",
                    "Increase min_samples_split to reduce tree complexity",
                    "Use n_jobs=-1 for parallel processing"
                ])
            elif samples_per_second < 500:
                recommendations.extend([
                    "Consider increasing n_jobs for better parallelization",
                    "Monitor tree depth and complexity"
                ])
            else:
                recommendations.append("Efficiency is good - current settings are optimal")
            
            return recommendations
            
        except:
            return ["Could not generate efficiency recommendations"]
    
    def _compare_with_random_forest(self):
        """Compare Extra Trees with Random Forest performance"""
        if not self.compare_with_random_forest:
            return
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Create comparable Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=min(50, self.n_estimators),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=1,
                bootstrap=True  # RF typically uses bootstrap
            )
            
            # Cross-validation comparison
            et_scores = cross_val_score(
                self.model_, self.X_original_, self.y_original_,
                cv=3, scoring='r2'
            )
            
            rf_scores = cross_val_score(
                rf_model, self.X_original_, self.y_original_,
                cv=3, scoring='r2'
            )
            
            # Performance comparison
            et_mean = np.mean(et_scores)
            rf_mean = np.mean(rf_scores)
            performance_difference = et_mean - rf_mean
            
            # Statistical significance test
            from scipy import stats
            try:
                t_stat, p_value = stats.ttest_rel(et_scores, rf_scores)
                significant_difference = p_value < 0.05
            except:
                significant_difference = None
                p_value = None
            
            # Comparison assessment
            comparison_assessment = self._assess_rf_comparison(
                et_mean, rf_mean, performance_difference, significant_difference
            )
            
            self.random_forest_comparison_ = {
                'extra_trees_scores': et_scores,
                'random_forest_scores': rf_scores,
                'extra_trees_mean': et_mean,
                'random_forest_mean': rf_mean,
                'performance_difference': performance_difference,
                'significant_difference': significant_difference,
                'p_value': p_value,
                'comparison_assessment': comparison_assessment
            }
            
        except Exception as e:
            self.random_forest_comparison_ = {
                'error': f'Could not compare with Random Forest: {str(e)}'
            }
    
    def _assess_rf_comparison(self, et_score, rf_score, difference, significant):
        """Assess Extra Trees vs Random Forest comparison"""
        try:
            if difference > 0.05:
                if significant:
                    performance_assessment = "Extra Trees significantly outperforms Random Forest"
                else:
                    performance_assessment = "Extra Trees appears better but not statistically significant"
            elif difference < -0.05:
                if significant:
                    performance_assessment = "Random Forest significantly outperforms Extra Trees"
                else:
                    performance_assessment = "Random Forest appears better but not statistically significant"
            else:
                performance_assessment = "Extra Trees and Random Forest have similar performance"
            
            # When to prefer Extra Trees
            et_advantages = [
                "Faster training due to random splits",
                "Better for high-dimensional data",
                "Less prone to overfitting with extreme randomness",
                "Good when interpretability is less critical"
            ]
            
            # When to prefer Random Forest
            rf_advantages = [
                "More interpretable individual trees",
                "Better for smaller datasets",
                "More established and well-understood",
                "Built-in out-of-bag evaluation"
            ]
            
            return {
                'performance_assessment': performance_assessment,
                'extra_trees_advantages': et_advantages,
                'random_forest_advantages': rf_advantages,
                'recommendation': self._get_algorithm_recommendation(difference, significant)
            }
            
        except:
            return {'performance_assessment': 'Unknown'}
    
    def _get_algorithm_recommendation(self, difference, significant):
        """Get recommendation on which algorithm to use"""
        try:
            if difference > 0.03 and significant:
                return "Prefer Extra Trees for this dataset - significant performance advantage"
            elif difference < -0.03 and significant:
                return "Prefer Random Forest for this dataset - significant performance advantage"
            elif difference > 0.01:
                return "Slight advantage to Extra Trees - consider speed vs interpretability tradeoffs"
            elif difference < -0.01:
                return "Slight advantage to Random Forest - consider stability vs speed tradeoffs"
            else:
                return "Similar performance - choose based on computational requirements and interpretability needs"
        except:
            return "Could not generate algorithm recommendation"
    
    def _analyze_randomness_impact(self):
        """Analyze the impact of extreme randomness on performance"""
        if not self.randomness_impact_analysis:
            return
        
        try:
            # Test different levels of randomness by varying max_features
            randomness_levels = ['sqrt', 'log2', None, 0.3, 0.7]
            randomness_results = []
            
            for max_feat in randomness_levels[:3]:  # Limit for efficiency
                try:
                    model = ExtraTreesRegressor(
                        n_estimators=30,
                        max_features=max_feat,
                        random_state=self.random_state,
                        n_jobs=1
                    )
                    
                    scores = cross_val_score(
                        model, self.X_original_, self.y_original_,
                        cv=3, scoring='r2'
                    )
                    
                    randomness_results.append({
                        'max_features': max_feat,
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'randomness_level': self._assess_randomness_level(max_feat)
                    })
                    
                except:
                    continue
            
            # Analyze randomness impact
            randomness_analysis = self._analyze_randomness_effects(randomness_results)
            
            self.randomness_impact_analysis_ = {
                'randomness_results': randomness_results,
                'randomness_analysis': randomness_analysis
            }
            
        except Exception as e:
            self.randomness_impact_analysis_ = {
                'error': f'Could not analyze randomness impact: {str(e)}'
            }
    
    def _assess_randomness_level(self, max_features):
        """Assess the level of randomness for given max_features"""
        if max_features is None:
            return "Low"
        elif max_features == 'sqrt':
            return "Medium"
        elif max_features == 'log2':
            return "High"
        elif isinstance(max_features, float) and max_features < 0.5:
            return "Very High"
        else:
            return "Medium"
    
    def _analyze_randomness_effects(self, results):
        """Analyze the effects of different randomness levels"""
        try:
            if len(results) < 2:
                return {'status': 'Insufficient data for randomness analysis'}
            
            # Find optimal randomness level
            best_result = max(results, key=lambda x: x['mean_score'])
            
            # Analyze variance vs performance tradeoff
            performances = [r['mean_score'] for r in results]
            variances = [r['std_score'] for r in results]
            
            # Correlation between randomness and performance
            randomness_scores = [3 if r['randomness_level'] == 'Very High' else 
                               2 if r['randomness_level'] == 'High' else
                               1 if r['randomness_level'] == 'Medium' else 0
                               for r in results]
            
            try:
                correlation = np.corrcoef(randomness_scores, performances)[0, 1]
            except:
                correlation = 0.0
            
            return {
                'optimal_randomness': best_result['max_features'],
                'optimal_randomness_level': best_result['randomness_level'],
                'best_performance': best_result['mean_score'],
                'randomness_performance_correlation': correlation,
                'interpretation': self._interpret_randomness_correlation(correlation)
            }
            
        except:
            return {'status': 'Error in randomness analysis'}
    
    def _interpret_randomness_correlation(self, correlation):
        """Interpret the correlation between randomness and performance"""
        if correlation > 0.5:
            return "Higher randomness improves performance - dataset benefits from extreme randomization"
        elif correlation > 0.2:
            return "Moderate positive effect of randomness - Extra Trees advantage confirmed"
        elif correlation > -0.2:
            return "Randomness has minimal impact - performance stable across settings"
        else:
            return "Higher randomness reduces performance - consider Random Forest instead"
    
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
            
            # Extra Trees performance
            if 'r2' in self.cross_validation_analysis_.get('cv_results', {}):
                et_performance = self.cross_validation_analysis_['cv_results']['r2']['mean']
            else:
                et_performance = self.model_.score(self.X_original_, self.y_original_)
            
            # Performance improvements
            improvements = {}
            for name, result in benchmark_results.items():
                if 'cv_r2_mean' in result:
                    improvement = et_performance - result['cv_r2_mean']
                    improvements[name] = {
                        'absolute_improvement': improvement,
                        'relative_improvement': improvement / max(0.01, abs(result['cv_r2_mean']))
                    }
            
            self.performance_benchmarking_ = {
                'extra_trees_r2': et_performance,
                'benchmark_results': benchmark_results,
                'improvements': improvements,
                'performance_ranking': self._rank_performance(et_performance, benchmark_results)
            }
            
        except Exception as e:
            self.performance_benchmarking_ = {
                'error': f'Could not analyze performance benchmarks: {str(e)}'
            }
    
    def _rank_performance(self, et_score, benchmark_results):
        """Rank Extra Trees performance against benchmarks"""
        try:
            all_scores = [et_score]
            model_names = ['Extra Trees']
            
            for name, result in benchmark_results.items():
                if 'cv_r2_mean' in result:
                    all_scores.append(result['cv_r2_mean'])
                    model_names.append(name.replace('_', ' ').title())
            
            # Sort by score
            sorted_indices = np.argsort(all_scores)[::-1]
            ranking = [(model_names[i], all_scores[i]) for i in sorted_indices]
            
            et_rank = next(i for i, (name, _) in enumerate(ranking) if name == 'Extra Trees') + 1
            
            return {
                'ranking': ranking,
                'extra_trees_rank': et_rank,
                'total_models': len(ranking),
                'performance_percentile': (len(ranking) - et_rank + 1) / len(ranking) * 100
            }
            
        except:
            return {'error': 'Could not rank performance'}
    
    def _analyze_split_quality(self):
        """Analyze quality of random splits"""
        if not self.split_quality_analysis:
            return
        
        try:
            # Sample trees for analysis
            sample_trees = self.model_.estimators_[:5]
            
            split_qualities = []
            
            for tree in sample_trees:
                tree_structure = tree.tree_
                
                # Extract split information
                features = tree_structure.feature[tree_structure.feature >= 0]
                thresholds = tree_structure.threshold[tree_structure.feature >= 0]
                impurities = tree_structure.impurity[tree_structure.feature >= 0]
                
                if len(features) > 0:
                    # Calculate split quality metrics
                    impurity_decreases = []
                    for i in range(len(features)):
                        left_child = tree_structure.children_left[i]
                        right_child = tree_structure.children_right[i]
                        
                        if left_child != -1 and right_child != -1:
                            left_impurity = tree_structure.impurity[left_child]
                            right_impurity = tree_structure.impurity[right_child]
                            left_samples = tree_structure.n_node_samples[left_child]
                            right_samples = tree_structure.n_node_samples[right_child]
                            total_samples = left_samples + right_samples
                            
                            weighted_child_impurity = (left_samples * left_impurity + 
                                                     right_samples * right_impurity) / total_samples
                            impurity_decrease = impurities[i] - weighted_child_impurity
                            impurity_decreases.append(impurity_decrease)
                    
                    split_qualities.append({
                        'mean_impurity_decrease': np.mean(impurity_decreases),
                        'std_impurity_decrease': np.std(impurity_decreases),
                        'n_splits': len(impurity_decreases)
                    })
            
            # Aggregate split quality metrics
            if split_qualities:
                split_quality_summary = {
                    'mean_impurity_decrease': np.mean([sq['mean_impurity_decrease'] for sq in split_qualities]),
                    'overall_std': np.mean([sq['std_impurity_decrease'] for sq in split_qualities]),
                    'consistency_score': 1.0 / (1.0 + np.std([sq['mean_impurity_decrease'] for sq in split_qualities])),
                    'trees_analyzed': len(split_qualities)
                }
                
                # Quality assessment
                quality_assessment = self._assess_split_quality(split_quality_summary)
                
                self.split_quality_analysis_ = {
                    'individual_trees': split_qualities,
                    'summary': split_quality_summary,
                    'quality_assessment': quality_assessment
                }
            else:
                self.split_quality_analysis_ = {
                    'error': 'No valid splits found for analysis'
                }
            
        except Exception as e:
            self.split_quality_analysis_ = {
                'error': f'Could not analyze split quality: {str(e)}'
            }
    
    def _assess_split_quality(self, summary):
        """Assess the quality of random splits"""
        try:
            consistency = summary['consistency_score']
            mean_decrease = summary['mean_impurity_decrease']
            
            if consistency > 0.8 and mean_decrease > 0.01:
                quality_level = "High - Random splits are effective"
            elif consistency > 0.6 and mean_decrease > 0.005:
                quality_level = "Good - Acceptable split quality"
            elif consistency > 0.4:
                quality_level = "Moderate - Random splits show some benefit"
            else:
                quality_level = "Low - Random splits may not be optimal for this data"
            
            return {
                'quality_level': quality_level,
                'consistency_score': consistency,
                'mean_impurity_decrease': mean_decrease,
                'interpretation': 'Higher consistency indicates stable random split benefits'
            }
            
        except:
            return {'quality_level': 'Unknown'}
    
    def _analyze_randomness_efficiency(self):
        """Analyze efficiency of randomness in Extra Trees"""
        if not self.randomness_efficiency_analysis:
            return
        
        try:
            # Compare random vs best split efficiency (conceptual analysis)
            # This is estimated since we can't directly compare with best splits
            
            # Analyze tree complexity resulting from random splits
            tree_complexities = []
            
            for tree in self.model_.estimators_[:10]:
                tree_structure = tree.tree_
                
                complexity_metrics = {
                    'depth': tree_structure.max_depth,
                    'nodes': tree_structure.node_count,
                    'leaves': tree_structure.n_leaves,
                    'depth_to_leaves_ratio': tree_structure.max_depth / tree_structure.n_leaves if tree_structure.n_leaves > 0 else 0
                }
                
                tree_complexities.append(complexity_metrics)
            
            # Aggregate complexity analysis
            complexity_summary = {
                'mean_depth': np.mean([tc['depth'] for tc in tree_complexities]),
                'mean_nodes': np.mean([tc['nodes'] for tc in tree_complexities]),
                'mean_leaves': np.mean([tc['leaves'] for tc in tree_complexities]),
                'complexity_variance': np.var([tc['nodes'] for tc in tree_complexities]),
                'efficiency_score': np.mean([tc['depth_to_leaves_ratio'] for tc in tree_complexities])
            }
            
            # Efficiency assessment
            efficiency_assessment = self._assess_randomness_efficiency(complexity_summary)
            
            self.randomness_efficiency_analysis_ = {
                'tree_complexities': tree_complexities,
                'complexity_summary': complexity_summary,
                'efficiency_assessment': efficiency_assessment
            }
            
        except Exception as e:
            self.randomness_efficiency_analysis_ = {
                'error': f'Could not analyze randomness efficiency: {str(e)}'
            }
    
    def _assess_randomness_efficiency(self, summary):
        """Assess efficiency of random splits"""
        try:
            efficiency_score = summary['efficiency_score']
            complexity_variance = summary['complexity_variance']
            
            if efficiency_score > 0.8 and complexity_variance > 100:
                efficiency_level = "High - Random splits create diverse, efficient trees"
            elif efficiency_score > 0.6:
                efficiency_level = "Good - Randomness provides reasonable efficiency"
            elif efficiency_score > 0.4:
                efficiency_level = "Moderate - Some efficiency from random splits"
            else:
                efficiency_level = "Low - Random splits may be creating inefficient trees"
            
            return {
                'efficiency_level': efficiency_level,
                'efficiency_score': efficiency_score,
                'complexity_variance': complexity_variance,
                'interpretation': 'Higher scores indicate better tree structure from random splits'
            }
            
        except:
            return {'efficiency_level': 'Unknown'}
    
    def _analyze_bias_variance(self):
        """Analyze bias-variance tradeoff in Extra Trees"""
        if not self.bias_variance_analysis:
            return
        
        try:
            # Simplified bias-variance analysis using bootstrap samples
            n_bootstrap = 20
            sample_size = min(200, len(self.X_original_))
            
            # Create test data for analysis
            test_indices = np.random.choice(len(self.X_original_), sample_size, replace=False)
            X_test = self.X_original_[test_indices]
            y_test = self.y_original_[test_indices]
            
            # Train multiple models on bootstrap samples
            predictions = []
            
            for i in range(n_bootstrap):
                # Bootstrap sample
                boot_indices = np.random.choice(len(self.X_original_), len(self.X_original_), replace=True)
                X_boot = self.X_original_[boot_indices]
                y_boot = self.y_original_[boot_indices]
                
                # Train model
                model = ExtraTreesRegressor(
                    n_estimators=30,
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    random_state=i,
                    n_jobs=1
                )
                
                model.fit(X_boot, y_boot)
                pred = model.predict(X_test)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate bias and variance
            mean_prediction = np.mean(predictions, axis=0)
            variance = np.mean(np.var(predictions, axis=0))
            bias_squared = np.mean((mean_prediction - y_test) ** 2)
            
            # Total error decomposition
            total_error = bias_squared + variance
            noise_estimate = np.var(y_test) * 0.1  # Rough noise estimate
            
            bias_variance_analysis = {
                'bias_squared': bias_squared,
                'variance': variance,
                'total_error': total_error,
                'noise_estimate': noise_estimate,
                'bias_variance_ratio': bias_squared / (variance + 1e-10),
                'variance_fraction': variance / total_error,
                'bias_fraction': bias_squared / total_error
            }
            
            # Assessment
            bv_assessment = self._assess_bias_variance(bias_variance_analysis)
            
            self.bias_variance_analysis_ = {
                'bias_variance_decomposition': bias_variance_analysis,
                'assessment': bv_assessment,
                'n_bootstrap_samples': n_bootstrap,
                'test_sample_size': sample_size
            }
            
        except Exception as e:
            self.bias_variance_analysis_ = {
                'error': f'Could not analyze bias-variance tradeoff: {str(e)}'
            }
    
    def _assess_bias_variance(self, analysis):
        """Assess bias-variance tradeoff"""
        try:
            bias_fraction = analysis['bias_fraction']
            variance_fraction = analysis['variance_fraction']
            ratio = analysis['bias_variance_ratio']
            
            if variance_fraction > 0.7:
                assessment = "High variance - model might benefit from more regularization"
                recommendation = "Consider reducing max_features or increasing min_samples_split"
            elif bias_fraction > 0.7:
                assessment = "High bias - model might be too simple"
                recommendation = "Consider increasing model complexity or reducing min_samples_leaf"
            elif 0.3 <= variance_fraction <= 0.7:
                assessment = "Good bias-variance balance"
                recommendation = "Current settings provide good tradeoff"
            else:
                assessment = "Moderate bias-variance tradeoff"
                recommendation = "Fine-tune regularization parameters"
            
            return {
                'assessment': assessment,
                'recommendation': recommendation,
                'bias_dominance': bias_fraction > variance_fraction,
                'variance_dominance': variance_fraction > bias_fraction,
                'balance_score': 1.0 - abs(bias_fraction - variance_fraction)
            }
            
        except:
            return {'assessment': 'Unknown bias-variance characteristics'}
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Ensemble Configuration", "Tree Parameters", "Extra Trees Settings", "Analysis Options", "Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Extra Trees Ensemble Configuration**")
            
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
                    help="Whether bootstrap samples are used (default False for Extra Trees)",
                    key=f"{key_prefix}_bootstrap"
                )
                
                n_jobs = st.selectbox(
                    "Parallel Jobs:",
                    options=[-1, 1, 2, 4],
                    index=[-1, 1, 2, 4].index(self.n_jobs) if self.n_jobs in [-1, 1, 2, 4] else 0,
                    help="Number of jobs for parallel processing (-1 = all cores)",
                    key=f"{key_prefix}_n_jobs"
                )
            
            with col2:
                max_samples = st.selectbox(
                    "Max Samples per Tree:",
                    options=[None, 0.7, 0.8, 0.9, 1.0],
                    index=0 if self.max_samples is None else [None, 0.7, 0.8, 0.9, 1.0].index(self.max_samples),
                    help="Maximum number of samples to draw for each tree",
                    key=f"{key_prefix}_max_samples"
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
                
                verbose = st.selectbox(
                    "Verbose Output:",
                    options=[0, 1, 2],
                    index=self.verbose,
                    help="Control the verbosity of training output",
                    key=f"{key_prefix}_verbose"
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
            st.markdown("**Extra Trees Specific Settings**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_features = st.selectbox(
                    "Max Features per Split:",
                    options=['sqrt', 'log2', None, 0.3, 0.5, 0.7],
                    index=['sqrt', 'log2', None, 0.3, 0.5, 0.7].index(self.max_features) if self.max_features in ['sqrt', 'log2', None, 0.3, 0.5, 0.7] else 0,
                    help="Number of features for random selection at each split",
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
                
                compare_with_random_forest = st.checkbox(
                    "Compare with Random Forest",
                    value=self.compare_with_random_forest,
                    help="Perform comparison analysis with Random Forest",
                    key=f"{key_prefix}_compare_with_random_forest"
                )
            
            with col2:
                st.markdown("**Extra Trees Advantages:**")
                st.info("""
                🎲 **Random Thresholds**: Uses random splits instead of optimal splits
                ⚡ **Faster Training**: No best split search required
                🎯 **Better Generalization**: Extreme randomness reduces overfitting
                📊 **High-Dimensional Data**: Excels with many features
                🔄 **Lower Variance**: More stable predictions than Random Forest
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
                
                split_randomness_analysis = st.checkbox(
                    "Split Randomness Analysis",
                    value=self.split_randomness_analysis,
                    help="Analyze impact of random splits",
                    key=f"{key_prefix}_split_randomness_analysis"
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
                
                randomness_impact_analysis = st.checkbox(
                    "Randomness Impact Analysis",
                    value=self.randomness_impact_analysis,
                    help="Analyze impact of different randomness levels",
                    key=f"{key_prefix}_randomness_impact_analysis"
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
                
                split_quality_analysis = st.checkbox(
                    "Split Quality Analysis",
                    value=self.split_quality_analysis,
                    help="Analyze quality of random splits",
                    key=f"{key_prefix}_split_quality_analysis"
                )
                
                bias_variance_analysis = st.checkbox(
                    "Bias-Variance Analysis",
                    value=self.bias_variance_analysis,
                    help="Analyze bias-variance tradeoff",
                    key=f"{key_prefix}_bias_variance_analysis"
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
            st.markdown("**Extra Trees Regressor - Extremely Randomized Trees**")
            
            # Algorithm information
            if st.button("📚 Algorithm Information", key=f"{key_prefix}_algo_info"):
                st.markdown("""
                **Extra Trees (Extremely Randomized Trees) - Advanced Ensemble Method**
                
                Extra Trees extends Random Forest by introducing extreme randomness in the tree 
                building process, using random thresholds for splits rather than searching for 
                optimal ones. This leads to faster training and often better generalization.
                
                **Core Principles:**
                • **Random Feature Selection** - Random subset of features at each split (like RF)
                • **Random Threshold Selection** - Random thresholds instead of optimal splits
                • **No Bootstrap** - Uses entire dataset by default (unlike Random Forest)
                • **Extreme Randomization** - Maximum randomness in tree construction
                • **Ensemble Averaging** - Final prediction is average of all tree predictions
                • **Faster Training** - No expensive best-split search required
                
                **Key Advantages:**
                • ⚡ **Faster Training** - Random splits eliminate best-split search
                • 🎯 **Better Generalization** - Extreme randomness reduces overfitting
                • 📊 **High-Dimensional Data** - Excels with many features
                • 🔄 **Lower Variance** - More stable than Random Forest
                • 🛡️ **Robust to Noise** - Random splits are less sensitive to outliers
                • 📈 **No Assumptions** - Handles non-linear relationships naturally
                """)
            
            # When to use Extra Trees
            if st.button("🎯 When to Use Extra Trees", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • High-dimensional datasets with many features
                • Need for fast training with good performance
                • Non-linear relationships in data
                • Overfitting issues with other tree methods
                • Computational efficiency is important
                
                **Data Characteristics:**
                • Large feature spaces (hundreds to thousands of features)
                • Moderate to large sample sizes
                • Noisy data or outliers present
                • Complex feature interactions
                • Time-sensitive training requirements
                
                **Examples:**
                • Genomics and bioinformatics (high-dimensional data)
                • Text mining and NLP feature extraction
                • Computer vision feature analysis
                • High-frequency financial modeling
                • Large-scale scientific simulations
                • Real-time recommendation systems
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Extremely fast training (faster than Random Forest)
                ✅ Often better generalization than Random Forest
                ✅ Excellent for high-dimensional data
                ✅ Built-in feature importance ranking
                ✅ Robust to overfitting due to extreme randomness
                ✅ Handles large feature spaces efficiently
                ✅ Lower variance than Random Forest
                ✅ No hyperparameter tuning for split selection
                ✅ Parallelizable training and prediction
                ✅ Good performance with default parameters
                
                **Limitations:**
                ❌ Individual trees are less interpretable (random splits)
                ❌ May underperform on small, simple datasets
                ❌ Random splits can miss obvious optimal splits
                ❌ Less control over individual tree quality
                ❌ May require more trees than Random Forest
                ❌ Can struggle with very structured data
                ❌ Less established than Random Forest
                """)
            
            # Extra Trees vs other methods
            if st.button("🔍 Extra Trees vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **Extra Trees vs Other Ensemble Methods:**
                
                **Extra Trees vs Random Forest:**
                • ET: Faster training, random thresholds, no bootstrap by default
                • RF: Optimal splits, bootstrap sampling, more interpretable trees
                • ET: Better for high-dimensional data, often superior generalization
                • RF: Better for smaller datasets, more predictable behavior
                
                **Extra Trees vs Gradient Boosting:**
                • ET: Parallel training, less prone to overfitting
                • GB: Sequential training, often higher accuracy but slower
                • ET: Better bias-variance tradeoff, more robust
                • GB: Better for structured data, requires careful tuning
                
                **Extra Trees vs Single Decision Tree:**
                • ET: Much better generalization, ensemble robustness
                • Tree: More interpretable, faster prediction per sample
                • ET: Handles complex patterns, feature importance
                • Tree: Simple rules, easier to explain decisions
                
                **Extra Trees vs Linear Models:**
                • ET: Handles non-linearity and interactions automatically
                • Linear: More interpretable, better extrapolation
                • ET: No assumptions about relationships
                • Linear: Faster training on large datasets
                
                **Extra Trees vs Neural Networks:**
                • ET: Better for tabular data, automatic feature selection
                • NN: Better for images, sequences, complex patterns
                • ET: Requires less data, easier to tune
                • NN: More flexible, can model any function
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Extra Trees Best Practices:**
                
                **Ensemble Configuration:**
                1. **n_estimators**: Start with 100, increase to 200-500 for better performance
                2. **max_features**: Use 'sqrt' for balanced randomness, 'log2' for more diversity
                3. **bootstrap=False**: Default setting leverages full dataset
                4. **max_samples**: Only relevant if bootstrap=True
                
                **Tree Parameters:**
                1. **max_depth**: Usually leave as None, use min_samples_split for control
                2. **min_samples_split**: 2-10 depending on dataset size
                3. **min_samples_leaf**: 1-3, higher values for more regularization
                4. **min_impurity_decrease**: Use sparingly, random splits are already regularized
                
                **Performance Optimization:**
                1. Use **n_jobs=-1** for parallel training
                2. Monitor convergence to determine optimal n_estimators
                3. Use feature importance for feature selection
                4. Compare with Random Forest to validate choice
                
                **Model Validation:**
                1. Use cross-validation for robust performance assessment
                2. Analyze randomness impact on your specific dataset
                3. Check tree diversity for ensemble quality
                4. Monitor computational efficiency vs accuracy tradeoff
                
                **When to Tune:**
                1. **max_features**: Critical parameter affecting randomness level
                2. **n_estimators**: More trees usually help until convergence
                3. **min_samples_split**: Controls tree complexity and speed
                4. **max_depth**: Only if default unlimited depth causes issues
                """)
            
            # Advanced techniques
            if st.button("🚀 Advanced Techniques", key=f"{key_prefix}_advanced"):
                st.markdown("""
                **Advanced Extra Trees Techniques:**
                
                **Randomness Optimization:**
                • **Adaptive Feature Selection**: Dynamically adjust max_features based on data
                • **Threshold Distribution Analysis**: Study optimal threshold distributions
                • **Multi-level Randomness**: Combine different randomness strategies
                • **Random Split Quality Assessment**: Evaluate effectiveness of random splits
                
                **Ensemble Enhancement:**
                • **Weighted Ensemble**: Weight trees based on individual performance
                • **Dynamic Ensemble Sizing**: Adaptive number of estimators
                • **Hybrid Ensembles**: Combine Extra Trees with other methods
                • **Selective Ensemble**: Choose best-performing trees post-training
                
                **Computational Optimization:**
                • **Progressive Training**: Add trees incrementally with early stopping
                • **Parallel Feature Engineering**: Optimize feature computation
                • **Memory Optimization**: Reduce memory footprint for large ensembles
                • **Incremental Learning**: Update ensemble with new data
                
                **Specialized Applications:**
                • **Time Series Ensembles**: Adapt for temporal data
                • **Multi-target Regression**: Handle multiple outputs simultaneously
                • **Online Learning**: Continuous model updates
                • **Quantile Regression**: Predict confidence intervals
                
                **Interpretation and Analysis:**
                • **Feature Interaction Discovery**: Use ensemble for feature engineering
                • **Uncertainty Quantification**: Leverage tree disagreement
                • **Model Distillation**: Create simpler models from ensemble
                • **Decision Path Analysis**: Understand ensemble decision patterns
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
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "compute_feature_importance": compute_feature_importance,
            "compute_permutation_importance": compute_permutation_importance,
            "split_randomness_analysis": split_randomness_analysis,
            "tree_diversity_analysis": tree_diversity_analysis,
            "ensemble_convergence_analysis": ensemble_convergence_analysis,
            "feature_interaction_analysis": feature_interaction_analysis,
            "learning_curve_analysis": learning_curve_analysis,
            "hyperparameter_sensitivity_analysis": hyperparameter_sensitivity_analysis,
            "computational_efficiency_analysis": computational_efficiency_analysis,
            "compare_with_random_forest": compare_with_random_forest,
            "randomness_impact_analysis": randomness_impact_analysis,
            "visualize_trees": visualize_trees,
            "max_trees_to_visualize": max_trees_to_visualize,
            "feature_importance_analysis": True,
            "prediction_distribution_analysis": True,
            "cross_validation_analysis": cross_validation_analysis,
            "cv_folds": cv_folds,
            "performance_benchmarking": performance_benchmarking,
            "split_quality_analysis": split_quality_analysis,
            "randomness_efficiency_analysis": True,
            "bias_variance_analysis": bias_variance_analysis
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return ExtraTreesRegressorPlugin(
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
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            random_state=hyperparameters.get("random_state", self.random_state),
            verbose=hyperparameters.get("verbose", self.verbose),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            ccp_alpha=hyperparameters.get("ccp_alpha", self.ccp_alpha),
            max_samples=hyperparameters.get("max_samples", self.max_samples),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            compute_permutation_importance=hyperparameters.get("compute_permutation_importance", self.compute_permutation_importance),
            split_randomness_analysis=hyperparameters.get("split_randomness_analysis", self.split_randomness_analysis),
            tree_diversity_analysis=hyperparameters.get("tree_diversity_analysis", self.tree_diversity_analysis),
            ensemble_convergence_analysis=hyperparameters.get("ensemble_convergence_analysis", self.ensemble_convergence_analysis),
            feature_interaction_analysis=hyperparameters.get("feature_interaction_analysis", self.feature_interaction_analysis),
            learning_curve_analysis=hyperparameters.get("learning_curve_analysis", self.learning_curve_analysis),
            hyperparameter_sensitivity_analysis=hyperparameters.get("hyperparameter_sensitivity_analysis", self.hyperparameter_sensitivity_analysis),
            computational_efficiency_analysis=hyperparameters.get("computational_efficiency_analysis", self.computational_efficiency_analysis),
            compare_with_random_forest=hyperparameters.get("compare_with_random_forest", self.compare_with_random_forest),
            randomness_impact_analysis=hyperparameters.get("randomness_impact_analysis", self.randomness_impact_analysis),
            visualize_trees=hyperparameters.get("visualize_trees", self.visualize_trees),
            max_trees_to_visualize=hyperparameters.get("max_trees_to_visualize", self.max_trees_to_visualize),
            feature_importance_analysis=hyperparameters.get("feature_importance_analysis", self.feature_importance_analysis),
            prediction_distribution_analysis=hyperparameters.get("prediction_distribution_analysis", self.prediction_distribution_analysis),
            cross_validation_analysis=hyperparameters.get("cross_validation_analysis", self.cross_validation_analysis),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            performance_benchmarking=hyperparameters.get("performance_benchmarking", self.performance_benchmarking),
            split_quality_analysis=hyperparameters.get("split_quality_analysis", self.split_quality_analysis),
            randomness_efficiency_analysis=hyperparameters.get("randomness_efficiency_analysis", self.randomness_efficiency_analysis),
            bias_variance_analysis=hyperparameters.get("bias_variance_analysis", self.bias_variance_analysis)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Extra Trees (minimal preprocessing needed)"""
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
        """Check if Extra Trees is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Extra Trees requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Extra Trees Regressor requires continuous numerical target values"
            
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
                advantages.append(f"Good sample size ({n_samples}) - adequate for Extra Trees ensemble")
            else:
                considerations.append(f"Small sample size ({n_samples}) - ensemble may have limited diversity")
            
            # Feature dimensionality assessment (Extra Trees excel with high dimensions)
            if n_features >= 100:
                advantages.append(f"High dimensionality ({n_features}) - ideal for Extra Trees' random feature selection")
            elif n_features >= 20:
                advantages.append(f"Moderate dimensionality ({n_features}) - good for Extra Trees randomness")
            else:
                considerations.append(f"Low dimensionality ({n_features}) - Extra Trees advantage may be limited")
            
            # Data characteristics favorable to Extra Trees
            advantages.append("Extremely randomized trees with superior generalization")
            advantages.append("Faster training than Random Forest due to random splits")
            advantages.append("Excellent for high-dimensional data and feature selection")
            advantages.append("Robust to overfitting through extreme randomness")
            
            # Check feature-to-sample ratio (Extra Trees handle high-dimensional data well)
            feature_sample_ratio = n_features / n_samples
            if feature_sample_ratio > 0.5:
                advantages.append(f"High feature-to-sample ratio ({feature_sample_ratio:.2f}) - Extra Trees excel in this scenario")
            elif feature_sample_ratio > 0.1:
                advantages.append(f"Moderate feature-to-sample ratio ({feature_sample_ratio:.2f}) - good for Extra Trees")
            else:
                considerations.append(f"Low feature-to-sample ratio ({feature_sample_ratio:.2f}) - consider simpler models")
            
            # Computational efficiency advantages
            if n_samples > 10000 or n_features > 100:
                advantages.append("Large dataset - benefits from Extra Trees' computational efficiency")
            
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
                f"🌳 Suitability for Extra Trees: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance with Extra Trees specific insights"""
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
        
        # Create feature importance dictionary with Extra Trees specific metrics
        feature_importance = {}
        for i, name in enumerate(self.feature_names_):
            feature_importance[name] = {
                'gini_importance': builtin_importance[i],
                'permutation_importance': permutation_importance[i] if permutation_importance is not None else None,
                'rank': np.where(feature_ranking == i)[0][0] + 1,
                'randomness_robustness': analysis['importance_stability']['randomness_robustness'][i],
                'stability_cv': analysis['importance_stability']['cv_importance'][i],
                'consensus': analysis['importance_stability']['consensus_importance'][i],
                'is_robust_important': analysis['robust_important_features'][i]
            }
        
        # Get top features
        top_features = analysis['top_features']
        
        return {
            'feature_importance': feature_importance,
            'top_features': [(f['name'], f['builtin_importance'], f.get('permutation_importance')) for f in top_features],
            'importance_statistics': analysis['importance_statistics'],
            'random_split_impact': analysis.get('random_split_impact', {}),
            'ensemble_info': {
                'n_trees': len(self.model_.estimators_),
                'randomness_stability': analysis['importance_statistics']['randomness_stability'],
                'robust_features_count': np.sum(analysis['robust_important_features'])
            },
            'interpretation': 'Extra Trees feature importance with randomness robustness analysis'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "algorithm": "Extra Trees Regressor",
            "ensemble_structure": {
                "n_estimators": len(self.model_.estimators_),
                "bootstrap": self.bootstrap,
                "max_features": self.max_features,
                "extreme_randomization": True
            },
            "tree_parameters": {
                "criterion": self.criterion,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "ccp_alpha": self.ccp_alpha
            },
            "randomness_analysis": {
                "split_randomness": self.split_randomness_analysis_.get('randomness_assessment', {}),
                "tree_diversity": self.tree_diversity_analysis_.get('diversity_assessment', {}),
                "computational_efficiency": self.computational_efficiency_analysis_.get('efficiency_assessment', {})
            },
            "interpretability": {
                "feature_importance_available": True,
                "randomness_robustness_analysis": True,
                "random_forest_comparison": bool(self.random_forest_comparison_),
                "split_quality_analysis": True
            }
        }
    
    def get_ensemble_analysis(self) -> Dict[str, Any]:
        """Get comprehensive ensemble analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        return {
            "feature_importance_analysis": self.feature_importance_analysis_,
            "split_randomness_analysis": self.split_randomness_analysis_,
            "tree_diversity_analysis": self.tree_diversity_analysis_,
            "ensemble_convergence_analysis": self.ensemble_convergence_analysis_,
            "feature_interaction_analysis": self.feature_interaction_analysis_,
            "learning_curve_analysis": self.learning_curve_analysis_,
            "hyperparameter_sensitivity_analysis": self.hyperparameter_sensitivity_analysis_,
            "computational_efficiency_analysis": self.computational_efficiency_analysis_,
            "random_forest_comparison": self.random_forest_comparison_,
            "randomness_impact_analysis": self.randomness_impact_analysis_,
            "cross_validation_analysis": self.cross_validation_analysis_,
            "performance_benchmarking": self.performance_benchmarking_,
            "split_quality_analysis": self.split_quality_analysis_,
            "randomness_efficiency_analysis": self.randomness_efficiency_analysis_,
            "bias_variance_analysis": self.bias_variance_analysis_
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Extra Trees Regressor",
            "type": "Extremely randomized trees ensemble with random threshold selection",
            "training_completed": True,
            "randomization_characteristics": {
                "extreme_randomization": True,
                "random_thresholds": True,
                "faster_than_random_forest": True,
                "better_generalization": True,
                "high_dimensional_optimized": True
            },
            "ensemble_structure": {
                "n_estimators": len(self.model_.estimators_),
                "bootstrap_sampling": self.bootstrap,
                "feature_randomness": self.max_features,
                "threshold_randomness": True
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
        if self.split_randomness_analysis_:
            info["randomness_analysis"] = {
                "randomness_level": self.split_randomness_analysis_.get('randomness_assessment', {}).get('randomness_level'),
                "diversity_score": self.split_randomness_analysis_.get('randomness_assessment', {}).get('diversity_score'),
                "overall_randomness_score": self.split_randomness_analysis_.get('randomness_assessment', {}).get('overall_randomness_score')
            }
        
        if self.random_forest_comparison_:
            info["random_forest_comparison"] = {
                "performance_assessment": self.random_forest_comparison_.get('comparison_assessment', {}).get('performance_assessment'),
                "extra_trees_advantage": self.random_forest_comparison_.get('performance_difference', 0) > 0
            }
        
        return info

    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Extra Trees Regressor-specific metrics based on the fitted model's
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
            A dictionary of Extra Trees Regressor-specific metrics.
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
                else:
                    return default
            return current if pd.notna(current) else default

        # --- Feature Importance Analysis ---
        if hasattr(self, 'feature_importance_analysis_') and self.feature_importance_analysis_:
            fi_analysis = self.feature_importance_analysis_
            if not fi_analysis.get('error'):
                metrics['fi_mean_importance'] = safe_get(fi_analysis, 'importance_statistics.mean_importance')
                metrics['fi_gini_coefficient'] = safe_get(fi_analysis, 'importance_statistics.gini_coefficient')
                metrics['fi_randomness_stability_mean'] = safe_get(fi_analysis, 'importance_statistics.randomness_stability')
                metrics['fi_top_5_concentration'] = safe_get(fi_analysis, 'importance_statistics.importance_concentration.top_5_concentration')
                metrics['fi_num_robust_features'] = np.sum(safe_get(fi_analysis, 'robust_important_features', []))
                metrics['fi_random_split_sensitivity_score'] = safe_get(fi_analysis, 'random_split_impact.randomness_sensitivity_score')

        # --- Split Randomness Analysis ---
        if hasattr(self, 'split_randomness_analysis_') and self.split_randomness_analysis_:
            sr_analysis = self.split_randomness_analysis_
            if not sr_analysis.get('error'):
                metrics['sr_mean_feature_diversity'] = safe_get(sr_analysis, 'randomness_metrics.mean_feature_diversity')
                metrics['sr_overall_randomness_score'] = safe_get(sr_analysis, 'randomness_assessment.overall_randomness_score')

        # --- Tree Diversity Analysis ---
        if hasattr(self, 'tree_diversity_analysis_') and self.tree_diversity_analysis_:
            td_analysis = self.tree_diversity_analysis_
            if not td_analysis.get('error'):
                metrics['td_prediction_diversity_score'] = safe_get(td_analysis, 'prediction_diversity.diversity_score')
                metrics['td_extreme_randomness_score'] = safe_get(td_analysis, 'extreme_diversity_metrics.extreme_randomness_score')
                metrics['td_mean_depth_cv'] = safe_get(td_analysis, 'structure_diversity.depth_diversity.cv')

        # --- Ensemble Convergence Analysis ---
        if hasattr(self, 'ensemble_convergence_analysis_') and self.ensemble_convergence_analysis_:
            ec_analysis = self.ensemble_convergence_analysis_
            if not ec_analysis.get('error'):
                metrics['ec_final_score_r2'] = safe_get(ec_analysis, 'convergence_analysis.final_score')
                metrics['ec_optimal_trees'] = safe_get(ec_analysis, 'convergence_analysis.optimal_trees')
                metrics['ec_fast_convergence_ratio'] = safe_get(ec_analysis, 'fast_convergence_analysis.fast_convergence_ratio')

        # --- Feature Interaction Analysis ---
        if hasattr(self, 'feature_interaction_analysis_') and self.feature_interaction_analysis_:
            fint_analysis = self.feature_interaction_analysis_
            if not fint_analysis.get('error') and fint_analysis.get('interactions'):
                strengths = [inter.get('interaction_strength', 0) for inter in fint_analysis['interactions']]
                metrics['fint_mean_top_interaction_strength'] = np.mean(strengths) if strengths else np.nan
                metrics['fint_mean_robustness'] = safe_get(fint_analysis, 'interaction_stability.stability_metrics.mean_robustness')

        # --- Learning Curve Analysis ---
        if hasattr(self, 'learning_curve_analysis_') and self.learning_curve_analysis_:
            lc_analysis = self.learning_curve_analysis_
            if not lc_analysis.get('error'):
                metrics['lc_final_generalization_gap'] = safe_get(lc_analysis, 'learning_analysis.final_gap')
                metrics['lc_best_val_score_r2'] = safe_get(lc_analysis, 'learning_analysis.best_val_score')
                metrics['lc_et_learning_efficiency_ratio'] = safe_get(lc_analysis, 'fast_learning_analysis.learning_efficiency_ratio')

        # --- Hyperparameter Sensitivity Analysis ---
        if hasattr(self, 'hyperparameter_sensitivity_analysis_') and self.hyperparameter_sensitivity_analysis_:
            hs_analysis = self.hyperparameter_sensitivity_analysis_
            if not hs_analysis.get('error'):
                metrics['hs_num_sensitive_params'] = len(safe_get(hs_analysis, 'sensitivity_assessment.sensitive_parameters', []))
                metrics['hs_max_features_score_range'] = safe_get(hs_analysis, 'sensitivity_results.max_features.score_range')

        # --- Computational Efficiency Analysis ---
        if hasattr(self, 'computational_efficiency_analysis_') and self.computational_efficiency_analysis_:
            ce_analysis = self.computational_efficiency_analysis_
            if not ce_analysis.get('error'):
                metrics['ce_training_time'] = safe_get(ce_analysis, 'training_time')
                metrics['ce_prediction_time_per_sample'] = safe_get(ce_analysis, 'prediction_time') / max(1, safe_get(ce_analysis, 'sample_size_tested', 1))
                metrics['ce_rf_speedup_training'] = safe_get(ce_analysis, 'rf_comparison.speedup_training')

        # --- Random Forest Comparison ---
        if hasattr(self, 'random_forest_comparison_') and self.random_forest_comparison_:
            rfcmp_analysis = self.random_forest_comparison_
            if not rfcmp_analysis.get('error'):
                metrics['rfcmp_et_mean_r2'] = safe_get(rfcmp_analysis, 'extra_trees_mean')
                metrics['rfcmp_rf_mean_r2'] = safe_get(rfcmp_analysis, 'random_forest_mean')
                metrics['rfcmp_performance_difference_r2'] = safe_get(rfcmp_analysis, 'performance_difference')

        # --- Randomness Impact Analysis ---
        if hasattr(self, 'randomness_impact_analysis_') and self.randomness_impact_analysis_:
            ri_analysis = self.randomness_impact_analysis_
            if not ri_analysis.get('error'):
                metrics['ri_best_performance_r2'] = safe_get(ri_analysis, 'randomness_analysis.best_performance')
                metrics['ri_randomness_performance_correlation'] = safe_get(ri_analysis, 'randomness_analysis.randomness_performance_correlation')

        # --- Cross-Validation Analysis ---
        if hasattr(self, 'cross_validation_analysis_') and self.cross_validation_analysis_:
            cv_analysis = self.cross_validation_analysis_
            if not cv_analysis.get('error'):
                metrics['cv_mean_r2'] = safe_get(cv_analysis, 'cv_results.r2.mean')
                metrics['cv_std_r2'] = safe_get(cv_analysis, 'cv_results.r2.std')
                metrics['cv_generalization_gap_r2'] = safe_get(cv_analysis, 'performance_assessment.generalization_gap')

        # --- Performance Benchmarking ---
        if hasattr(self, 'performance_benchmarking_') and self.performance_benchmarking_:
            bm_analysis = self.performance_benchmarking_
            if not bm_analysis.get('error'):
                metrics['bm_et_r2'] = safe_get(bm_analysis, 'extra_trees_r2')
                metrics['bm_improvement_over_mean_predictor_r2'] = safe_get(bm_analysis, 'improvements.mean_predictor.absolute_improvement')
                metrics['bm_et_rank'] = safe_get(bm_analysis, 'performance_ranking.extra_trees_rank')

        # --- Split Quality Analysis ---
        if hasattr(self, 'split_quality_analysis_') and self.split_quality_analysis_:
            sq_analysis = self.split_quality_analysis_
            if not sq_analysis.get('error'):
                metrics['sq_mean_impurity_decrease'] = safe_get(sq_analysis, 'summary.mean_impurity_decrease')
                metrics['sq_consistency_score'] = safe_get(sq_analysis, 'summary.consistency_score')

        # --- Randomness Efficiency Analysis ---
        if hasattr(self, 'randomness_efficiency_analysis_') and self.randomness_efficiency_analysis_:
            reff_analysis = self.randomness_efficiency_analysis_
            if not reff_analysis.get('error'):
                metrics['reff_efficiency_score'] = safe_get(reff_analysis, 'complexity_summary.efficiency_score') # depth_to_leaves_ratio
                metrics['reff_complexity_variance'] = safe_get(reff_analysis, 'complexity_summary.complexity_variance')

        # --- Bias-Variance Analysis ---
        if hasattr(self, 'bias_variance_analysis_') and self.bias_variance_analysis_:
            bv_analysis = self.bias_variance_analysis_
            if not bv_analysis.get('error'):
                metrics['bv_bias_squared'] = safe_get(bv_analysis, 'bias_variance_decomposition.bias_squared')
                metrics['bv_variance'] = safe_get(bv_analysis, 'bias_variance_decomposition.variance')
                metrics['bv_bias_variance_ratio'] = safe_get(bv_analysis, 'bias_variance_decomposition.bias_variance_ratio')
                metrics['bv_balance_score'] = safe_get(bv_analysis, 'assessment.balance_score')

        # Remove NaN or None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if pd.notna(v)}

        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return ExtraTreesRegressorPlugin()
