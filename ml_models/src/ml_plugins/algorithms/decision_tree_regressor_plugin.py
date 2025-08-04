import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
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


class DecisionTreeRegressorPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Decision Tree Regressor Plugin - Non-linear, Interpretable Regression
    
    This plugin implements Decision Tree Regression, providing a non-parametric approach
    that can capture complex non-linear relationships while maintaining high interpretability.
    Decision trees recursively partition the feature space to minimize prediction error,
    creating a tree structure that can be easily visualized and understood.
    
    Key Features:
    - Non-linear relationship modeling without assumptions
    - High interpretability with tree visualization
    - Feature importance analysis
    - Automatic handling of mixed data types
    - No requirement for data normalization
    - Robust to outliers
    - Built-in feature selection through splitting criteria
    - Comprehensive tree analysis and pruning strategies
    - Advanced interpretability tools and visualizations
    """
    
    def __init__(
        self,
        # Core tree parameters
        criterion='squared_error',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        
        # Regularization parameters
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        
        # Control parameters
        random_state=42,
        ccp_alpha=0.0,
        
        # Analysis options
        compute_feature_importance=True,
        compute_permutation_importance=True,
        tree_interpretation_analysis=True,
        decision_path_analysis=True,
        pruning_analysis=True,
        
        # Visualization options
        visualize_tree=True,
        max_tree_depth_display=5,
        feature_importance_analysis=True,
        leaf_analysis=True,
        
        # Advanced analysis
        overfitting_analysis=True,
        complexity_analysis=True,
        stability_analysis=True,
        
        # Performance options
        cross_validation_analysis=True,
        cv_folds=5
    ):
        super().__init__()
        
        # Core tree parameters
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        
        # Regularization parameters
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        
        # Control parameters
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        
        # Analysis options
        self.compute_feature_importance = compute_feature_importance
        self.compute_permutation_importance = compute_permutation_importance
        self.tree_interpretation_analysis = tree_interpretation_analysis
        self.decision_path_analysis = decision_path_analysis
        self.pruning_analysis = pruning_analysis
        
        # Visualization options
        self.visualize_tree = visualize_tree
        self.max_tree_depth_display = max_tree_depth_display
        self.feature_importance_analysis = feature_importance_analysis
        self.leaf_analysis = leaf_analysis
        
        # Advanced analysis
        self.overfitting_analysis = overfitting_analysis
        self.complexity_analysis = complexity_analysis
        self.stability_analysis = stability_analysis
        
        # Performance options
        self.cross_validation_analysis = cross_validation_analysis
        self.cv_folds = cv_folds
        
        # Required plugin metadata
        self._name = "Decision Tree Regressor"
        self._description = "Non-linear, interpretable regression using decision trees"
        self._category = "Tree-Based Models"
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 10
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        
        # Analysis results
        self.feature_importance_analysis_ = {}
        self.tree_analysis_ = {}
        self.decision_path_analysis_ = {}
        self.pruning_analysis_ = {}
        self.overfitting_analysis_ = {}
        self.complexity_analysis_ = {}
        self.stability_analysis_ = {}
        self.cross_validation_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Decision Tree Regressor with comprehensive analysis
        
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
        
        # Create and configure Decision Tree model
        self.model_ = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha
        )
        
        # Fit the model
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        # Perform comprehensive analysis
        self._analyze_feature_importance()
        self._analyze_tree_structure()
        self._analyze_decision_paths()
        self._analyze_pruning_effects()
        self._analyze_overfitting()
        self._analyze_complexity()
        self._analyze_stability()
        self._analyze_cross_validation()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted Decision Tree
        
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
    
    def predict_with_paths(self, X):
        """
        Make predictions and return decision paths
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        
        Returns:
        --------
        results : dict
            Dictionary containing predictions and decision paths
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Get predictions
        y_pred = self.model_.predict(X)
        
        # Get decision paths
        decision_paths = self.model_.decision_path(X)
        leaf_ids = self.model_.apply(X)
        
        # Get path explanations for each prediction
        path_explanations = []
        for i in range(X.shape[0]):
            explanation = self._explain_prediction_path(X[i], decision_paths[i], leaf_ids[i])
            path_explanations.append(explanation)
        
        return {
            'predictions': y_pred,
            'decision_paths': decision_paths,
            'leaf_ids': leaf_ids,
            'path_explanations': path_explanations
        }
    
    def _explain_prediction_path(self, x_sample, decision_path, leaf_id):
        """Explain the decision path for a single prediction"""
        try:
            # Get the path to the leaf
            path_nodes = decision_path.indices
            
            explanation = {
                'path_length': len(path_nodes),
                'leaf_id': leaf_id,
                'decisions': [],
                'final_prediction': self.model_.tree_.value[leaf_id][0][0]
            }
            
            tree = self.model_.tree_
            
            for node_id in path_nodes[:-1]:  # Exclude leaf node
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = self.feature_names_[feature_idx]
                feature_value = x_sample[feature_idx]
                
                # Determine direction
                if feature_value <= threshold:
                    direction = "≤"
                    next_node = tree.children_left[node_id]
                else:
                    direction = ">"
                    next_node = tree.children_right[node_id]
                
                decision = {
                    'node_id': node_id,
                    'feature': feature_name,
                    'feature_index': feature_idx,
                    'threshold': threshold,
                    'feature_value': feature_value,
                    'direction': direction,
                    'condition': f"{feature_name} {direction} {threshold:.3f}",
                    'samples_at_node': tree.n_node_samples[node_id],
                    'impurity_at_node': tree.impurity[node_id]
                }
                
                explanation['decisions'].append(decision)
            
            return explanation
            
        except Exception as e:
            return {'error': f'Could not explain path: {str(e)}'}
    
    def _analyze_feature_importance(self):
        """Analyze feature importance using multiple methods"""
        if not self.compute_feature_importance:
            return
        
        try:
            # Built-in feature importance (Gini/entropy-based)
            builtin_importance = self.model_.feature_importances_
            
            # Permutation importance if requested
            permutation_imp = None
            if self.compute_permutation_importance:
                try:
                    perm_imp_result = permutation_importance(
                        self.model_, self.X_original_, self.y_original_,
                        n_repeats=10, random_state=self.random_state, scoring='neg_mean_squared_error'
                    )
                    permutation_imp = perm_imp_result.importances_mean
                    permutation_imp_std = perm_imp_result.importances_std
                except:
                    permutation_imp = None
                    permutation_imp_std = None
            
            # Feature importance ranking
            importance_ranking = np.argsort(builtin_importance)[::-1]
            
            # Calculate importance statistics
            importance_stats = {
                'mean_importance': np.mean(builtin_importance),
                'std_importance': np.std(builtin_importance),
                'max_importance': np.max(builtin_importance),
                'min_importance': np.min(builtin_importance),
                'importance_concentration': np.sum(builtin_importance[:5]) if len(builtin_importance) >= 5 else np.sum(builtin_importance),
                'gini_coefficient': self._calculate_gini_coefficient(builtin_importance)
            }
            
            # Identify important features
            importance_threshold = np.mean(builtin_importance) + np.std(builtin_importance)
            important_features = builtin_importance > importance_threshold
            
            self.feature_importance_analysis_ = {
                'builtin_importance': builtin_importance,
                'permutation_importance': permutation_imp,
                'permutation_importance_std': permutation_imp_std if permutation_imp is not None else None,
                'feature_ranking': importance_ranking,
                'feature_names': self.feature_names_,
                'importance_statistics': importance_stats,
                'important_features': important_features,
                'importance_threshold': importance_threshold,
                'top_features': [
                    (self.feature_names_[i], builtin_importance[i], 
                     permutation_imp[i] if permutation_imp is not None else None)
                    for i in importance_ranking[:10]
                ]
            }
            
        except Exception as e:
            self.feature_importance_analysis_ = {
                'error': f'Could not analyze feature importance: {str(e)}'
            }
    
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for importance concentration"""
        try:
            sorted_values = np.sort(values)
            n = len(values)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        except:
            return 0.0
    
    def _analyze_tree_structure(self):
        """Analyze the structure and properties of the fitted tree"""
        if not self.tree_interpretation_analysis:
            return
        
        try:
            tree = self.model_.tree_
            
            # Basic tree statistics
            n_nodes = tree.node_count
            n_leaves = tree.n_leaves
            max_depth = tree.max_depth
            
            # Calculate tree complexity metrics
            tree_complexity = {
                'n_nodes': n_nodes,
                'n_leaves': n_leaves,
                'max_depth': max_depth,
                'average_depth': self._calculate_average_depth(),
                'balance_ratio': self._calculate_tree_balance(),
                'leaf_purity': self._calculate_leaf_purity(),
                'tree_complexity_score': n_leaves * max_depth  # Simple complexity measure
            }
            
            # Analyze node statistics
            node_stats = self._analyze_node_statistics()
            
            # Analyze splits
            split_analysis = self._analyze_splits()
            
            # Tree interpretation
            tree_interpretation = {
                'complexity_assessment': self._assess_tree_complexity(tree_complexity),
                'interpretability_score': self._calculate_interpretability_score(tree_complexity),
                'overfitting_risk': self._assess_overfitting_risk(tree_complexity)
            }
            
            self.tree_analysis_ = {
                'tree_complexity': tree_complexity,
                'node_statistics': node_stats,
                'split_analysis': split_analysis,
                'tree_interpretation': tree_interpretation,
                'tree_rules': self._extract_tree_rules() if n_leaves <= 50 else "Too complex for rule extraction"
            }
            
        except Exception as e:
            self.tree_analysis_ = {
                'error': f'Could not analyze tree structure: {str(e)}'
            }
    
    def _calculate_average_depth(self):
        """Calculate average depth of leaves weighted by samples"""
        try:
            tree = self.model_.tree_
            
            def get_depth(node_id, current_depth=0):
                if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf
                    return [(current_depth, tree.n_node_samples[node_id])]
                else:
                    depths = []
                    depths.extend(get_depth(tree.children_left[node_id], current_depth + 1))
                    depths.extend(get_depth(tree.children_right[node_id], current_depth + 1))
                    return depths
            
            depth_samples = get_depth(0)
            total_samples = sum(samples for _, samples in depth_samples)
            weighted_depth = sum(depth * samples for depth, samples in depth_samples) / total_samples
            
            return weighted_depth
            
        except:
            return self.model_.tree_.max_depth
    
    def _calculate_tree_balance(self):
        """Calculate tree balance ratio"""
        try:
            tree = self.model_.tree_
            
            def calculate_balance(node_id):
                if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf
                    return 1.0
                
                left_samples = tree.n_node_samples[tree.children_left[node_id]]
                right_samples = tree.n_node_samples[tree.children_right[node_id]]
                
                if left_samples == 0 or right_samples == 0:
                    return 0.0
                
                balance = min(left_samples, right_samples) / max(left_samples, right_samples)
                return balance
            
            # Calculate balance for all internal nodes
            balances = []
            for node_id in range(tree.node_count):
                if tree.children_left[node_id] != tree.children_right[node_id]:  # Internal node
                    balances.append(calculate_balance(node_id))
            
            return np.mean(balances) if balances else 1.0
            
        except:
            return 0.5
    
    def _calculate_leaf_purity(self):
        """Calculate average leaf purity (inverse of impurity)"""
        try:
            tree = self.model_.tree_
            
            leaf_impurities = []
            leaf_samples = []
            
            for node_id in range(tree.node_count):
                if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf
                    leaf_impurities.append(tree.impurity[node_id])
                    leaf_samples.append(tree.n_node_samples[node_id])
            
            # Weighted average purity (1 - impurity)
            total_samples = sum(leaf_samples)
            weighted_impurity = sum(imp * samples for imp, samples in zip(leaf_impurities, leaf_samples)) / total_samples
            
            return 1.0 - weighted_impurity
            
        except:
            return 0.5
    
    def _analyze_node_statistics(self):
        """Analyze statistics of tree nodes"""
        try:
            tree = self.model_.tree_
            
            # Collect node information
            node_depths = []
            node_samples = []
            node_impurities = []
            
            def traverse(node_id, depth=0):
                node_depths.append(depth)
                node_samples.append(tree.n_node_samples[node_id])
                node_impurities.append(tree.impurity[node_id])
                
                if tree.children_left[node_id] != tree.children_right[node_id]:  # Internal node
                    traverse(tree.children_left[node_id], depth + 1)
                    traverse(tree.children_right[node_id], depth + 1)
            
            traverse(0)
            
            return {
                'mean_node_samples': np.mean(node_samples),
                'std_node_samples': np.std(node_samples),
                'mean_node_impurity': np.mean(node_impurities),
                'std_node_impurity': np.std(node_impurities),
                'depth_distribution': {
                    'mean': np.mean(node_depths),
                    'std': np.std(node_depths),
                    'max': np.max(node_depths)
                }
            }
            
        except Exception as e:
            return {'error': f'Could not analyze node statistics: {str(e)}'}
    
    def _analyze_splits(self):
        """Analyze the quality and distribution of splits"""
        try:
            tree = self.model_.tree_
            
            # Collect split information
            split_features = []
            split_thresholds = []
            split_impurity_decreases = []
            
            for node_id in range(tree.node_count):
                if tree.children_left[node_id] != tree.children_right[node_id]:  # Internal node
                    split_features.append(tree.feature[node_id])
                    split_thresholds.append(tree.threshold[node_id])
                    
                    # Calculate impurity decrease
                    parent_impurity = tree.impurity[node_id]
                    left_samples = tree.n_node_samples[tree.children_left[node_id]]
                    right_samples = tree.n_node_samples[tree.children_right[node_id]]
                    total_samples = tree.n_node_samples[node_id]
                    
                    left_impurity = tree.impurity[tree.children_left[node_id]]
                    right_impurity = tree.impurity[tree.children_right[node_id]]
                    
                    weighted_child_impurity = (left_samples * left_impurity + right_samples * right_impurity) / total_samples
                    impurity_decrease = parent_impurity - weighted_child_impurity
                    split_impurity_decreases.append(impurity_decrease)
            
            # Analyze feature usage
            feature_usage = np.bincount(split_features, minlength=self.n_features_in_)
            feature_usage_freq = feature_usage / np.sum(feature_usage)
            
            return {
                'total_splits': len(split_features),
                'feature_usage_count': feature_usage,
                'feature_usage_frequency': feature_usage_freq,
                'most_used_features': np.argsort(feature_usage)[::-1][:5],
                'split_quality': {
                    'mean_impurity_decrease': np.mean(split_impurity_decreases),
                    'std_impurity_decrease': np.std(split_impurity_decreases),
                    'max_impurity_decrease': np.max(split_impurity_decreases),
                    'min_impurity_decrease': np.min(split_impurity_decreases)
                },
                'threshold_statistics': {
                    'mean_threshold': np.mean(split_thresholds),
                    'std_threshold': np.std(split_thresholds),
                    'threshold_range': np.max(split_thresholds) - np.min(split_thresholds)
                }
            }
            
        except Exception as e:
            return {'error': f'Could not analyze splits: {str(e)}'}
    
    def _assess_tree_complexity(self, complexity_metrics):
        """Assess overall tree complexity"""
        try:
            n_nodes = complexity_metrics['n_nodes']
            n_leaves = complexity_metrics['n_leaves']
            max_depth = complexity_metrics['max_depth']
            
            # Define complexity categories
            if n_leaves <= 5 and max_depth <= 3:
                return "Very Simple - Highly interpretable"
            elif n_leaves <= 15 and max_depth <= 5:
                return "Simple - Good interpretability"
            elif n_leaves <= 50 and max_depth <= 8:
                return "Moderate - Fair interpretability"
            elif n_leaves <= 100 and max_depth <= 12:
                return "Complex - Limited interpretability"
            else:
                return "Very Complex - Poor interpretability"
                
        except:
            return "Unknown complexity"
    
    def _calculate_interpretability_score(self, complexity_metrics):
        """Calculate interpretability score (0-1, higher is more interpretable)"""
        try:
            n_leaves = complexity_metrics['n_leaves']
            max_depth = complexity_metrics['max_depth']
            
            # Normalize factors (lower is better for interpretability)
            leaf_penalty = min(1.0, n_leaves / 50.0)  # Penalty increases with leaves
            depth_penalty = min(1.0, max_depth / 10.0)  # Penalty increases with depth
            
            # Calculate interpretability score
            interpretability = 1.0 - (0.6 * leaf_penalty + 0.4 * depth_penalty)
            return max(0.0, interpretability)
            
        except:
            return 0.5
    
    def _assess_overfitting_risk(self, complexity_metrics):
        """Assess risk of overfitting based on tree complexity"""
        try:
            n_leaves = complexity_metrics['n_leaves']
            n_samples = len(self.y_original_)
            
            samples_per_leaf = n_samples / n_leaves
            
            if samples_per_leaf < 2:
                return "Very High - Likely overfitting"
            elif samples_per_leaf < 5:
                return "High - Monitor carefully"
            elif samples_per_leaf < 10:
                return "Moderate - Consider pruning"
            elif samples_per_leaf < 20:
                return "Low - Good balance"
            else:
                return "Very Low - May be underfitting"
                
        except:
            return "Unknown risk"
    
    def _extract_tree_rules(self, max_rules=20):
        """Extract human-readable rules from the tree"""
        try:
            if self.model_.tree_.n_leaves > max_rules:
                return f"Tree too complex for rule extraction ({self.model_.tree_.n_leaves} leaves > {max_rules})"
            
            # Use sklearn's export_text for rule extraction
            rules_text = export_text(
                self.model_,
                feature_names=self.feature_names_,
                max_depth=self.max_tree_depth_display
            )
            
            return rules_text
            
        except Exception as e:
            return f"Could not extract rules: {str(e)}"
    
    def _analyze_decision_paths(self):
        """Analyze decision paths for training data"""
        if not self.decision_path_analysis:
            return
        
        try:
            # Get decision paths for training data
            decision_paths = self.model_.decision_path(self.X_original_)
            leaf_ids = self.model_.apply(self.X_original_)
            
            # Analyze path statistics
            path_lengths = np.array([len(decision_paths[i].indices) for i in range(len(self.X_original_))])
            
            # Analyze leaf distribution
            unique_leaves, leaf_counts = np.unique(leaf_ids, return_counts=True)
            
            path_analysis = {
                'mean_path_length': np.mean(path_lengths),
                'std_path_length': np.std(path_lengths),
                'min_path_length': np.min(path_lengths),
                'max_path_length': np.max(path_lengths),
                'path_length_distribution': {
                    'percentile_25': np.percentile(path_lengths, 25),
                    'percentile_50': np.percentile(path_lengths, 50),
                    'percentile_75': np.percentile(path_lengths, 75)
                }
            }
            
            leaf_distribution = {
                'n_unique_leaves': len(unique_leaves),
                'mean_samples_per_leaf': np.mean(leaf_counts),
                'std_samples_per_leaf': np.std(leaf_counts),
                'max_samples_in_leaf': np.max(leaf_counts),
                'min_samples_in_leaf': np.min(leaf_counts),
                'leaf_utilization': len(unique_leaves) / self.model_.tree_.n_leaves
            }
            
            self.decision_path_analysis_ = {
                'path_analysis': path_analysis,
                'leaf_distribution': leaf_distribution,
                'decision_paths_matrix': decision_paths,
                'leaf_ids': leaf_ids
            }
            
        except Exception as e:
            self.decision_path_analysis_ = {
                'error': f'Could not analyze decision paths: {str(e)}'
            }
    
    def _analyze_pruning_effects(self):
        """Analyze the effects of different pruning strategies"""
        if not self.pruning_analysis:
            return
        
        try:
            # Cost complexity pruning path
            path = self.model_.cost_complexity_pruning_path(self.X_original_, self.y_original_)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            
            # Train trees with different alpha values
            tree_performance = []
            tree_complexity = []
            
            for ccp_alpha in ccp_alphas[::max(1, len(ccp_alphas)//10)]:  # Sample alphas
                try:
                    tree = DecisionTreeRegressor(
                        criterion=self.criterion,
                        random_state=self.random_state,
                        ccp_alpha=ccp_alpha
                    )
                    tree.fit(self.X_original_, self.y_original_)
                    
                    # Evaluate performance
                    y_pred = tree.predict(self.X_original_)
                    mse = mean_squared_error(self.y_original_, y_pred)
                    r2 = r2_score(self.y_original_, y_pred)
                    
                    tree_performance.append({
                        'ccp_alpha': ccp_alpha,
                        'mse': mse,
                        'r2': r2,
                        'n_leaves': tree.tree_.n_leaves,
                        'max_depth': tree.tree_.max_depth
                    })
                    
                    tree_complexity.append({
                        'ccp_alpha': ccp_alpha,
                        'n_nodes': tree.tree_.node_count,
                        'n_leaves': tree.tree_.n_leaves,
                        'max_depth': tree.tree_.max_depth
                    })
                except:
                    continue
            
            # Find optimal alpha (best trade-off)
            optimal_alpha_idx = self._find_optimal_alpha(tree_performance)
            optimal_alpha = tree_performance[optimal_alpha_idx]['ccp_alpha'] if optimal_alpha_idx is not None else None
            
            self.pruning_analysis_ = {
                'ccp_alphas': ccp_alphas,
                'impurities': impurities,
                'tree_performance': tree_performance,
                'tree_complexity': tree_complexity,
                'optimal_alpha': optimal_alpha,
                'current_alpha': self.ccp_alpha,
                'pruning_recommendation': self._get_pruning_recommendation(optimal_alpha)
            }
            
        except Exception as e:
            self.pruning_analysis_ = {
                'error': f'Could not analyze pruning effects: {str(e)}'
            }
    
    def _find_optimal_alpha(self, performance_data):
        """Find optimal alpha for pruning"""
        try:
            if len(performance_data) < 2:
                return None
            
            # Calculate efficiency (R² per leaf)
            efficiencies = []
            for perf in performance_data:
                if perf['n_leaves'] > 0:
                    efficiency = perf['r2'] / perf['n_leaves']
                    efficiencies.append(efficiency)
                else:
                    efficiencies.append(0)
            
            # Find alpha with best efficiency
            optimal_idx = np.argmax(efficiencies)
            return optimal_idx
            
        except:
            return None
    
    def _get_pruning_recommendation(self, optimal_alpha):
        """Get pruning recommendation based on analysis"""
        try:
            current_alpha = self.ccp_alpha
            
            if optimal_alpha is None:
                return "Could not determine optimal pruning"
            
            if optimal_alpha > current_alpha * 1.5:
                return f"Consider more aggressive pruning (α={optimal_alpha:.6f})"
            elif optimal_alpha < current_alpha * 0.5:
                return f"Consider less aggressive pruning (α={optimal_alpha:.6f})"
            else:
                return "Current pruning level is near optimal"
                
        except:
            return "Could not generate pruning recommendation"
    
    def _analyze_overfitting(self):
        """Analyze potential overfitting issues"""
        if not self.overfitting_analysis:
            return
        
        try:
            # Training performance
            y_pred_train = self.model_.predict(self.X_original_)
            train_mse = mean_squared_error(self.y_original_, y_pred_train)
            train_r2 = r2_score(self.y_original_, y_pred_train)
            
            # Simple validation: split training data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_original_, self.y_original_, test_size=0.2, random_state=self.random_state
            )
            
            # Train new model on subset
            val_model = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                ccp_alpha=self.ccp_alpha
            )
            val_model.fit(X_train, y_train)
            
            y_pred_val = val_model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            
            # Calculate overfitting metrics
            mse_gap = val_mse - train_mse
            r2_gap = train_r2 - val_r2
            
            # Overfitting assessment
            overfitting_assessment = self._assess_overfitting_level(mse_gap, r2_gap, train_r2, val_r2)
            
            self.overfitting_analysis_ = {
                'training_performance': {
                    'mse': train_mse,
                    'r2': train_r2
                },
                'validation_performance': {
                    'mse': val_mse,
                    'r2': val_r2
                },
                'performance_gaps': {
                    'mse_gap': mse_gap,
                    'r2_gap': r2_gap,
                    'relative_mse_increase': mse_gap / train_mse if train_mse > 0 else float('inf'),
                    'relative_r2_decrease': r2_gap / train_r2 if train_r2 > 0 else float('inf')
                },
                'overfitting_assessment': overfitting_assessment,
                'recommendations': self._get_overfitting_recommendations(overfitting_assessment)
            }
            
        except Exception as e:
            self.overfitting_analysis_ = {
                'error': f'Could not analyze overfitting: {str(e)}'
            }
    
    def _assess_overfitting_level(self, mse_gap, r2_gap, train_r2, val_r2):
        """Assess the level of overfitting"""
        try:
            # Multiple criteria for overfitting assessment
            criteria = []
            
            # R² gap criterion
            if r2_gap > 0.2:
                criteria.append("Large R² gap")
            elif r2_gap > 0.1:
                criteria.append("Moderate R² gap")
            
            # Validation R² criterion
            if val_r2 < 0:
                criteria.append("Negative validation R²")
            elif val_r2 < 0.3:
                criteria.append("Low validation R²")
            
            # Training R² criterion (too perfect)
            if train_r2 > 0.99:
                criteria.append("Suspiciously high training R²")
            
            # Overall assessment
            if len(criteria) >= 2:
                level = "High overfitting risk"
            elif len(criteria) == 1:
                level = "Moderate overfitting risk"
            else:
                level = "Low overfitting risk"
            
            return {
                'level': level,
                'criteria_met': criteria,
                'r2_gap': r2_gap,
                'validation_r2': val_r2,
                'training_r2': train_r2
            }
            
        except:
            return {'level': 'Unknown', 'criteria_met': []}
    
    def _get_overfitting_recommendations(self, assessment):
        """Get recommendations to reduce overfitting"""
        try:
            level = assessment['level']
            recommendations = []
            
            if 'High' in level:
                recommendations.extend([
                    "Increase min_samples_split (try 10-20)",
                    "Increase min_samples_leaf (try 5-10)",
                    "Reduce max_depth (try 3-7)",
                    "Increase ccp_alpha for pruning (try 0.01-0.1)",
                    "Consider ensemble methods (Random Forest)"
                ])
            elif 'Moderate' in level:
                recommendations.extend([
                    "Slightly increase min_samples_split",
                    "Consider mild pruning (ccp_alpha=0.001-0.01)",
                    "Reduce max_depth if very high"
                ])
            else:
                recommendations.append("Current regularization appears adequate")
            
            return recommendations
            
        except:
            return ["Could not generate recommendations"]
    
    def _analyze_complexity(self):
        """Analyze model complexity and generalization"""
        if not self.complexity_analysis:
            return
        
        try:
            # Tree structure complexity
            tree = self.model_.tree_
            structure_complexity = {
                'nodes_to_samples_ratio': tree.node_count / len(self.y_original_),
                'leaves_to_samples_ratio': tree.n_leaves / len(self.y_original_),
                'depth_to_features_ratio': tree.max_depth / self.n_features_in_,
                'average_samples_per_leaf': len(self.y_original_) / tree.n_leaves
            }
            
            # Model capacity analysis
            capacity_analysis = {
                'theoretical_capacity': 2 ** tree.max_depth,  # Maximum possible leaves
                'actual_leaves': tree.n_leaves,
                'capacity_utilization': tree.n_leaves / (2 ** tree.max_depth),
                'effective_parameters': tree.n_leaves  # Each leaf is a parameter
            }
            
            # Complexity assessment
            complexity_score = self._calculate_complexity_score(structure_complexity, capacity_analysis)
            complexity_interpretation = self._interpret_complexity(complexity_score)
            
            self.complexity_analysis_ = {
                'structure_complexity': structure_complexity,
                'capacity_analysis': capacity_analysis,
                'complexity_score': complexity_score,
                'complexity_interpretation': complexity_interpretation,
                'generalization_assessment': self._assess_generalization_ability(structure_complexity)
            }
            
        except Exception as e:
            self.complexity_analysis_ = {
                'error': f'Could not analyze complexity: {str(e)}'
            }
    
    def _calculate_complexity_score(self, structure, capacity):
        """Calculate overall complexity score (0-1, higher is more complex)"""
        try:
            # Normalize components
            node_ratio = min(1.0, structure['nodes_to_samples_ratio'] * 10)  # Scale to 0-1
            leaf_ratio = min(1.0, structure['leaves_to_samples_ratio'] * 20)  # Scale to 0-1
            depth_ratio = min(1.0, structure['depth_to_features_ratio'])
            capacity_util = capacity['capacity_utilization']
            
            # Weighted combination
            complexity = (0.3 * node_ratio + 0.3 * leaf_ratio + 0.2 * depth_ratio + 0.2 * capacity_util)
            return min(1.0, complexity)
            
        except:
            return 0.5
    
    def _interpret_complexity(self, score):
        """Interpret complexity score"""
        if score < 0.2:
            return "Very Low - May be underfitting"
        elif score < 0.4:
            return "Low - Good for generalization"
        elif score < 0.6:
            return "Moderate - Balanced complexity"
        elif score < 0.8:
            return "High - Risk of overfitting"
        else:
            return "Very High - Likely overfitting"
    
    def _assess_generalization_ability(self, structure):
        """Assess the model's generalization ability"""
        try:
            avg_samples_per_leaf = structure['average_samples_per_leaf']
            
            if avg_samples_per_leaf >= 20:
                return "Excellent - High confidence in generalization"
            elif avg_samples_per_leaf >= 10:
                return "Good - Should generalize well"
            elif avg_samples_per_leaf >= 5:
                return "Fair - Moderate generalization expected"
            elif avg_samples_per_leaf >= 2:
                return "Poor - Limited generalization ability"
            else:
                return "Very Poor - Likely memorizing training data"
                
        except:
            return "Unknown generalization ability"
    
    def _analyze_stability(self):
        """Analyze model stability across different random seeds"""
        if not self.stability_analysis:
            return
        
        try:
            # Train multiple models with different random seeds
            n_models = 5
            models = []
            performances = []
            
            for seed in range(n_models):
                model = DecisionTreeRegressor(
                    criterion=self.criterion,
                    splitter=self.splitter,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=seed,
                    ccp_alpha=self.ccp_alpha
                )
                
                model.fit(self.X_original_, self.y_original_)
                models.append(model)
                
                # Evaluate performance
                y_pred = model.predict(self.X_original_)
                r2 = r2_score(self.y_original_, y_pred)
                mse = mean_squared_error(self.y_original_, y_pred)
                
                performances.append({
                    'r2': r2,
                    'mse': mse,
                    'n_leaves': model.tree_.n_leaves,
                    'max_depth': model.tree_.max_depth
                })
            
            # Analyze stability metrics
            r2_scores = [p['r2'] for p in performances]
            mse_scores = [p['mse'] for p in performances]
            n_leaves = [p['n_leaves'] for p in performances]
            
            stability_metrics = {
                'r2_stability': {
                    'mean': np.mean(r2_scores),
                    'std': np.std(r2_scores),
                    'min': np.min(r2_scores),
                    'max': np.max(r2_scores),
                    'range': np.max(r2_scores) - np.min(r2_scores)
                },
                'mse_stability': {
                    'mean': np.mean(mse_scores),
                    'std': np.std(mse_scores),
                    'coefficient_of_variation': np.std(mse_scores) / np.mean(mse_scores)
                },
                'structure_stability': {
                    'leaf_count_std': np.std(n_leaves),
                    'leaf_count_range': np.max(n_leaves) - np.min(n_leaves),
                    'mean_leaves': np.mean(n_leaves)
                }
            }
            
            # Stability assessment
            stability_assessment = self._assess_stability(stability_metrics)
            
            self.stability_analysis_ = {
                'stability_metrics': stability_metrics,
                'individual_performances': performances,
                'stability_assessment': stability_assessment,
                'recommendations': self._get_stability_recommendations(stability_assessment)
            }
            
        except Exception as e:
            self.stability_analysis_ = {
                'error': f'Could not analyze stability: {str(e)}'
            }
    
    def _assess_stability(self, metrics):
        """Assess overall model stability"""
        try:
            r2_cv = metrics['r2_stability']['std'] / abs(metrics['r2_stability']['mean'])
            structure_cv = metrics['structure_stability']['leaf_count_std'] / metrics['structure_stability']['mean_leaves']
            
            # Stability categories
            if r2_cv < 0.05 and structure_cv < 0.2:
                level = "Very Stable"
            elif r2_cv < 0.1 and structure_cv < 0.4:
                level = "Stable"
            elif r2_cv < 0.2 and structure_cv < 0.6:
                level = "Moderately Stable"
            else:
                level = "Unstable"
            
            return {
                'level': level,
                'r2_coefficient_of_variation': r2_cv,
                'structure_coefficient_of_variation': structure_cv,
                'r2_range': metrics['r2_stability']['range']
            }
            
        except:
            return {'level': 'Unknown', 'r2_coefficient_of_variation': None}
    
    def _get_stability_recommendations(self, assessment):
        """Get recommendations to improve stability"""
        try:
            level = assessment['level']
            recommendations = []
            
            if level == "Unstable":
                recommendations.extend([
                    "Increase min_samples_split for more stable splits",
                    "Increase min_samples_leaf to reduce variance",
                    "Consider ensemble methods (Random Forest)",
                    "Use cross-validation for model selection"
                ])
            elif level == "Moderately Stable":
                recommendations.extend([
                    "Consider slight increase in regularization",
                    "Monitor performance on validation set"
                ])
            else:
                recommendations.append("Model shows good stability")
            
            return recommendations
            
        except:
            return ["Could not generate stability recommendations"]
    
    def _analyze_cross_validation(self):
        """Perform cross-validation analysis"""
        if not self.cross_validation_analysis:
            return
        
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(
                self.model_, self.X_original_, self.y_original_,
                cv=self.cv_folds, scoring='r2'
            )
            
            cv_mse_scores = -cross_val_score(
                self.model_, self.X_original_, self.y_original_,
                cv=self.cv_folds, scoring='neg_mean_squared_error'
            )
            
            # Training score for comparison
            train_score = self.model_.score(self.X_original_, self.y_original_)
            
            cv_analysis = {
                'cv_r2_scores': cv_scores,
                'cv_mse_scores': cv_mse_scores,
                'cv_r2_mean': np.mean(cv_scores),
                'cv_r2_std': np.std(cv_scores),
                'cv_mse_mean': np.mean(cv_mse_scores),
                'cv_mse_std': np.std(cv_mse_scores),
                'train_r2': train_score,
                'generalization_gap': train_score - np.mean(cv_scores),
                'cv_confidence_interval': {
                    'lower': np.mean(cv_scores) - 1.96 * np.std(cv_scores),
                    'upper': np.mean(cv_scores) + 1.96 * np.std(cv_scores)
                }
            }
            
            # Performance assessment
            performance_assessment = self._assess_cv_performance(cv_analysis)
            
            self.cross_validation_analysis_ = {
                'cv_analysis': cv_analysis,
                'performance_assessment': performance_assessment,
                'cv_folds': self.cv_folds
            }
            
        except Exception as e:
            self.cross_validation_analysis_ = {
                'error': f'Could not perform cross-validation analysis: {str(e)}'
            }
    
    def _assess_cv_performance(self, cv_analysis):
        """Assess cross-validation performance"""
        try:
            cv_mean = cv_analysis['cv_r2_mean']
            cv_std = cv_analysis['cv_r2_std']
            gap = cv_analysis['generalization_gap']
            
            # Performance categories
            if cv_mean > 0.8 and gap < 0.1:
                assessment = "Excellent - High performance with good generalization"
            elif cv_mean > 0.6 and gap < 0.2:
                assessment = "Good - Solid performance with acceptable generalization"
            elif cv_mean > 0.4 and gap < 0.3:
                assessment = "Fair - Moderate performance, monitor overfitting"
            elif cv_mean > 0.2:
                assessment = "Poor - Low performance, consider model complexity"
            else:
                assessment = "Very Poor - Model not learning effectively"
            
            return {
                'overall_assessment': assessment,
                'cv_mean_r2': cv_mean,
                'cv_std_r2': cv_std,
                'generalization_gap': gap,
                'reliability': 'High' if cv_std < 0.1 else 'Medium' if cv_std < 0.2 else 'Low'
            }
            
        except:
            return {'overall_assessment': 'Unknown', 'reliability': 'Unknown'}
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Tree Structure", "Regularization", "Analysis Options", "Visualization", "Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Decision Tree Structure Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                criterion = st.selectbox(
                    "Split Criterion:",
                    options=['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    index=['squared_error', 'friedman_mse', 'absolute_error', 'poisson'].index(self.criterion),
                    help="Function to measure split quality",
                    key=f"{key_prefix}_criterion"
                )
                
                splitter = st.selectbox(
                    "Split Strategy:",
                    options=['best', 'random'],
                    index=['best', 'random'].index(self.splitter),
                    help="Strategy to choose split at each node",
                    key=f"{key_prefix}_splitter"
                )
                
                max_depth = st.selectbox(
                    "Maximum Depth:",
                    options=[None, 3, 5, 7, 10, 15, 20],
                    index=0 if self.max_depth is None else [None, 3, 5, 7, 10, 15, 20].index(self.max_depth),
                    help="Maximum depth of the tree (None = unlimited)",
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
            
            with col2:
                min_samples_leaf = st.number_input(
                    "Min Samples Leaf:",
                    value=self.min_samples_leaf,
                    min_value=1,
                    max_value=20,
                    step=1,
                    help="Minimum samples required at leaf node",
                    key=f"{key_prefix}_min_samples_leaf"
                )
                
                max_features = st.selectbox(
                    "Max Features:",
                    options=[None, 'sqrt', 'log2', 'auto'],
                    index=0 if self.max_features is None else [None, 'sqrt', 'log2', 'auto'].index(self.max_features),
                    help="Number of features to consider for best split",
                    key=f"{key_prefix}_max_features"
                )
                
                max_leaf_nodes = st.selectbox(
                    "Max Leaf Nodes:",
                    options=[None, 10, 20, 50, 100, 200],
                    index=0 if self.max_leaf_nodes is None else [None, 10, 20, 50, 100, 200].index(self.max_leaf_nodes),
                    help="Maximum number of leaf nodes",
                    key=f"{key_prefix}_max_leaf_nodes"
                )
                
                random_state = st.number_input(
                    "Random Seed:",
                    value=int(self.random_state),
                    min_value=0,
                    max_value=1000,
                    help="For reproducible results",
                    key=f"{key_prefix}_random_state"
                )
        
        with tab2:
            st.markdown("**Regularization and Pruning**")
            
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
                min_weight_fraction_leaf = st.number_input(
                    "Min Weight Fraction Leaf:",
                    value=self.min_weight_fraction_leaf,
                    min_value=0.0,
                    max_value=0.5,
                    step=0.01,
                    help="Minimum weighted fraction of input samples at leaf",
                    key=f"{key_prefix}_min_weight_fraction_leaf"
                )
                
                st.markdown("**Regularization Guide:**")
                st.info("""
                • **min_samples_split**: Prevents splits with few samples
                • **min_samples_leaf**: Ensures leaves have enough samples
                • **ccp_alpha**: Post-pruning complexity parameter
                • **min_impurity_decrease**: Pre-pruning threshold
                """)
        
        with tab3:
            st.markdown("**Analysis Options**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                compute_feature_importance = st.checkbox(
                    "Feature Importance Analysis",
                    value=self.compute_feature_importance,
                    help="Compute built-in feature importance",
                    key=f"{key_prefix}_compute_feature_importance"
                )
                
                compute_permutation_importance = st.checkbox(
                    "Permutation Importance",
                    value=self.compute_permutation_importance,
                    help="Compute permutation-based feature importance",
                    key=f"{key_prefix}_compute_permutation_importance"
                )
                
                tree_interpretation_analysis = st.checkbox(
                    "Tree Structure Analysis",
                    value=self.tree_interpretation_analysis,
                    help="Analyze tree structure and complexity",
                    key=f"{key_prefix}_tree_interpretation_analysis"
                )
                
                decision_path_analysis = st.checkbox(
                    "Decision Path Analysis",
                    value=self.decision_path_analysis,
                    help="Analyze decision paths through tree",
                    key=f"{key_prefix}_decision_path_analysis"
                )
                
                pruning_analysis = st.checkbox(
                    "Pruning Analysis",
                    value=self.pruning_analysis,
                    help="Analyze cost complexity pruning effects",
                    key=f"{key_prefix}_pruning_analysis"
                )
            
            with col2:
                overfitting_analysis = st.checkbox(
                    "Overfitting Analysis",
                    value=self.overfitting_analysis,
                    help="Analyze overfitting risk",
                    key=f"{key_prefix}_overfitting_analysis"
                )
                
                complexity_analysis = st.checkbox(
                    "Complexity Analysis",
                    value=self.complexity_analysis,
                    help="Analyze model complexity metrics",
                    key=f"{key_prefix}_complexity_analysis"
                )
                
                stability_analysis = st.checkbox(
                    "Stability Analysis",
                    value=self.stability_analysis,
                    help="Analyze model stability across random seeds",
                    key=f"{key_prefix}_stability_analysis"
                )
                
                cross_validation_analysis = st.checkbox(
                    "Cross-Validation Analysis",
                    value=self.cross_validation_analysis,
                    help="Perform cross-validation analysis",
                    key=f"{key_prefix}_cross_validation_analysis"
                )
                
                if cross_validation_analysis:
                    cv_folds = st.number_input(
                        "CV Folds:",
                        value=self.cv_folds,
                        min_value=3,
                        max_value=10,
                        step=1,
                        key=f"{key_prefix}_cv_folds"
                    )
                else:
                    cv_folds = self.cv_folds
        
        with tab4:
            st.markdown("**Visualization Options**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                visualize_tree = st.checkbox(
                    "Tree Visualization",
                    value=self.visualize_tree,
                    help="Generate tree structure plots",
                    key=f"{key_prefix}_visualize_tree"
                )
                
                max_tree_depth_display = st.number_input(
                    "Max Depth for Display:",
                    value=self.max_tree_depth_display,
                    min_value=3,
                    max_value=10,
                    step=1,
                    help="Maximum depth to show in tree visualization",
                    key=f"{key_prefix}_max_tree_depth_display"
                )
                
                feature_importance_analysis = st.checkbox(
                    "Feature Importance Plots",
                    value=self.feature_importance_analysis,
                    help="Generate feature importance visualizations",
                    key=f"{key_prefix}_feature_importance_analysis"
                )
            
            with col2:
                leaf_analysis = st.checkbox(
                    "Leaf Analysis",
                    value=self.leaf_analysis,
                    help="Analyze leaf node characteristics",
                    key=f"{key_prefix}_leaf_analysis"
                )
                
                st.markdown("**Visualization Tips:**")
                st.info("""
                • Tree plots work best for small trees (depth ≤ 5)
                • Feature importance helps identify key variables
                • Leaf analysis shows prediction distribution
                • Decision paths explain individual predictions
                """)
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Decision Tree Regressor** - Non-linear, Interpretable Regression:
            • 🌳 Recursive binary splits to minimize prediction error
            • 📊 No assumptions about data distribution
            • 🔍 **Automatic Feature Selection** - Uses most informative features for splits
            • 🎯 **Rule-based Predictions** - Generates interpretable if-then rules
            • 📈 **Handles Non-linearity** - Captures complex patterns naturally
            • 🛡️ **Robust to Outliers** - Splits based on thresholds, not distances
            • ⚡ **No Data Preprocessing** - Works with raw features (no scaling needed)
            
            **Mathematical Foundation:**
            • Recursive binary splitting: minimize prediction error at each split
            • Split criterion: squared_error, friedman_mse, absolute_error
            • Impurity measures: MSE, MAE for regression
            • Stopping criteria: min_samples_split, max_depth, min_impurity_decrease
            • Pruning: Cost complexity pruning with α parameter
            """)
            
            # When to use Decision Trees
            if st.button("🎯 When to Use Decision Trees", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • Need interpretable models (explainable AI)
                • Non-linear relationships suspected
                • Mixed data types (numerical + categorical)
                • Feature interactions important
                • Rule-based decision making desired
                
                **Data Characteristics:**
                • Medium-sized datasets (100s to 10,000s of samples)
                • Mixed feature types without preprocessing needs
                • Robust to outliers required
                • Feature scaling not feasible/desired
                • Hierarchical decision structure makes sense
                
                **Examples:**
                • Medical diagnosis with decision rules
                • Credit scoring with interpretable criteria
                • Customer segmentation with clear rules
                • Quality control with threshold-based decisions
                • Regulatory compliance requiring explainable models
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Highly interpretable (can visualize and explain decisions)
                ✅ No assumptions about data distribution
                ✅ Handles non-linear relationships naturally
                ✅ Automatic feature selection through splits
                ✅ No need for data preprocessing (scaling, normalization)
                ✅ Robust to outliers
                ✅ Fast training and prediction
                ✅ Can handle missing values (with modifications)
                ✅ Feature interactions captured automatically
                
                **Limitations:**
                ❌ Prone to overfitting (especially deep trees)
                ❌ High variance (small data changes → different trees)
                ❌ Biased toward features with more levels
                ❌ Cannot extrapolate beyond training data range
                ❌ Instability (small changes can cause big differences)
                ❌ May create overly complex trees
                ❌ Limited expressiveness for smooth functions
                """)
            
            # Tree vs other methods
            if st.button("🔍 Trees vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **Decision Trees vs Other Regression Methods:**
                
                **Trees vs Linear Regression:**
                • Trees: Non-linear, no assumptions, interpretable rules
                • Linear: Parametric, assumes linearity, coefficient interpretation
                • Trees: Better for interactions and non-linearity
                • Linear: Better for smooth relationships and extrapolation
                
                **Trees vs Random Forest:**
                • Trees: Single model, highly interpretable
                • Random Forest: Ensemble, better performance, less interpretable
                • Trees: Prone to overfitting
                • Random Forest: More robust, reduces variance
                
                **Trees vs Neural Networks:**
                • Trees: Highly interpretable, rule-based
                • Neural Networks: Black box, continuous functions
                • Trees: Better for tabular data with clear decision boundaries
                • Neural Networks: Better for complex patterns, images, text
                
                **Trees vs SVM:**
                • Trees: No kernel choice, interpretable
                • SVM: Kernel flexibility, mathematical foundation
                • Trees: Better for categorical features
                • SVM: Better for high-dimensional continuous data
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Decision Tree Best Practices:**
                
                **Preventing Overfitting:**
                1. **Pre-pruning**: Set max_depth (5-10), min_samples_split (10-20)
                2. **Post-pruning**: Use ccp_alpha (0.001-0.01) for complexity pruning
                3. **Validation**: Always use cross-validation or holdout validation
                4. **Sample requirements**: Ensure min_samples_leaf ≥ 5-10
                
                **Hyperparameter Tuning:**
                1. Start with max_depth=5, min_samples_split=20
                2. Use grid search for: max_depth, min_samples_split, ccp_alpha
                3. Monitor training vs validation performance
                4. Prefer simpler trees when performance is similar
                
                **Feature Engineering:**
                1. Trees handle mixed types well - minimal preprocessing needed
                2. Consider binning continuous features if interpretability is key
                3. Feature interactions are captured automatically
                4. Remove irrelevant features to improve interpretability
                
                **Model Validation:**
                1. Use cross-validation to assess generalization
                2. Plot learning curves to detect overfitting
                3. Analyze feature importance for insights
                4. Validate tree rules make domain sense
                
                **Interpretation:**
                1. Visualize tree structure (for small trees)
                2. Extract and validate decision rules
                3. Analyze feature importance rankings
                4. Test individual prediction paths
                """)
            
            # Advanced usage
            if st.button("🚀 Advanced Techniques", key=f"{key_prefix}_advanced"):
                st.markdown("""
                **Advanced Decision Tree Techniques:**
                
                **Ensemble Methods:**
                • **Random Forest**: Multiple trees with randomness
                • **Gradient Boosting**: Sequential error correction
                • **Extra Trees**: Extremely randomized trees
                • **Stacking**: Combine with other algorithms
                
                **Pruning Strategies:**
                • **Cost Complexity**: Balance tree size vs accuracy
                • **Reduced Error**: Remove nodes that don't improve validation
                • **Minimum Description Length**: Information theory-based
                • **Cross-validation**: Prune based on CV performance
                
                **Splitting Improvements:**
                • **Multivariate Splits**: Linear combinations of features
                • **Oblique Trees**: Non-axis-parallel splits
                • **Fuzzy Trees**: Soft splitting boundaries
                • **Evolutionary Trees**: Genetic algorithm optimization
                
                **Interpretability Enhancements:**
                • **Rule Extraction**: Convert to if-then rules
                • **Path Analysis**: Trace individual predictions
                • **Feature Interaction**: Analyze two-way interactions
                • **Surrogate Splits**: Handle missing values interpretably
                
                **Handling Imbalanced Data:**
                • **Sample Weights**: Weight classes differently
                • **Balanced Splitting**: Consider class balance in splits
                • **Cost-sensitive**: Different misclassification costs
                • **Ensemble Methods**: Combine with resampling
                """)
        
        return {
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "random_state": random_state,
            "ccp_alpha": ccp_alpha,
            "compute_feature_importance": compute_feature_importance,
            "compute_permutation_importance": compute_permutation_importance,
            "tree_interpretation_analysis": tree_interpretation_analysis,
            "decision_path_analysis": decision_path_analysis,
            "pruning_analysis": pruning_analysis,
            "visualize_tree": visualize_tree,
            "max_tree_depth_display": max_tree_depth_display,
            "feature_importance_analysis": feature_importance_analysis,
            "leaf_analysis": leaf_analysis,
            "overfitting_analysis": overfitting_analysis,
            "complexity_analysis": complexity_analysis,
            "stability_analysis": stability_analysis,
            "cross_validation_analysis": cross_validation_analysis,
            "cv_folds": cv_folds
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return DecisionTreeRegressorPlugin(
            criterion=hyperparameters.get("criterion", self.criterion),
            splitter=hyperparameters.get("splitter", self.splitter),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            min_samples_split=hyperparameters.get("min_samples_split", self.min_samples_split),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", self.min_samples_leaf),
            min_weight_fraction_leaf=hyperparameters.get("min_weight_fraction_leaf", self.min_weight_fraction_leaf),
            max_features=hyperparameters.get("max_features", self.max_features),
            max_leaf_nodes=hyperparameters.get("max_leaf_nodes", self.max_leaf_nodes),
            min_impurity_decrease=hyperparameters.get("min_impurity_decrease", self.min_impurity_decrease),
            random_state=hyperparameters.get("random_state", self.random_state),
            ccp_alpha=hyperparameters.get("ccp_alpha", self.ccp_alpha),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            compute_permutation_importance=hyperparameters.get("compute_permutation_importance", self.compute_permutation_importance),
            tree_interpretation_analysis=hyperparameters.get("tree_interpretation_analysis", self.tree_interpretation_analysis),
            decision_path_analysis=hyperparameters.get("decision_path_analysis", self.decision_path_analysis),
            pruning_analysis=hyperparameters.get("pruning_analysis", self.pruning_analysis),
            visualize_tree=hyperparameters.get("visualize_tree", self.visualize_tree),
            max_tree_depth_display=hyperparameters.get("max_tree_depth_display", self.max_tree_depth_display),
            feature_importance_analysis=hyperparameters.get("feature_importance_analysis", self.feature_importance_analysis),
            leaf_analysis=hyperparameters.get("leaf_analysis", self.leaf_analysis),
            overfitting_analysis=hyperparameters.get("overfitting_analysis", self.overfitting_analysis),
            complexity_analysis=hyperparameters.get("complexity_analysis", self.complexity_analysis),
            stability_analysis=hyperparameters.get("stability_analysis", self.stability_analysis),
            cross_validation_analysis=hyperparameters.get("cross_validation_analysis", self.cross_validation_analysis),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Decision Tree (minimal preprocessing needed)"""
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
        """Check if Decision Tree is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Decision Tree requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Decision Tree Regressor requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # Sample size assessment
            if n_samples >= 1000:
                advantages.append(f"Large sample size ({n_samples}) - good for stable trees")
            elif n_samples >= 100:
                advantages.append(f"Good sample size ({n_samples}) - adequate for tree building")
            else:
                considerations.append(f"Small sample size ({n_samples}) - risk of overfitting, use regularization")
            
            # Feature dimensionality assessment
            if n_features <= 20:
                advantages.append(f"Moderate dimensionality ({n_features}) - good interpretability")
            elif n_features <= 100:
                considerations.append(f"High dimensionality ({n_features}) - may need feature selection")
            else:
                considerations.append(f"Very high dimensionality ({n_features}) - consider ensemble methods")
            
            # Data characteristics
            advantages.append("No data preprocessing required (scaling, normalization)")
            advantages.append("Handles non-linear relationships naturally")
            advantages.append("Robust to outliers")
            advantages.append("Highly interpretable with decision rules")
            
            # Check for potential issues
            unique_values = np.unique(y)
            if len(unique_values) < 5:
                considerations.append(f"Few unique target values ({len(unique_values)}) - consider classification")
            
            # Samples per feature ratio
            ratio = n_samples / n_features
            if ratio < 5:
                considerations.append(f"Low samples-to-features ratio ({ratio:.1f}) - high overfitting risk")
            elif ratio < 20:
                considerations.append(f"Moderate samples-to-features ratio ({ratio:.1f}) - use regularization")
            else:
                advantages.append(f"Good samples-to-features ratio ({ratio:.1f})")
            
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
                f"📊 Suitability for Decision Trees: {suitability}"
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
        permutation_importance = analysis['permutation_importance']
        feature_ranking = analysis['feature_ranking']
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, name in enumerate(self.feature_names_):
            feature_importance[name] = {
                'gini_importance': builtin_importance[i],
                'permutation_importance': permutation_importance[i] if permutation_importance is not None else None,
                'rank': np.where(feature_ranking == i)[0][0] + 1,
                'is_important': analysis['important_features'][i]
            }
        
        # Get top features
        top_features = analysis['top_features']
        
        return {
            'feature_importance': feature_importance,
            'top_features': top_features,
            'importance_statistics': analysis['importance_statistics'],
            'tree_info': {
                'n_nodes': self.model_.tree_.node_count,
                'n_leaves': self.model_.tree_.n_leaves,
                'max_depth': self.model_.tree_.max_depth,
                'interpretable': True
            },
            'interpretation': 'Tree-based feature importance (Gini impurity decrease)'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        tree = self.model_.tree_
        
        return {
            "algorithm": "Decision Tree Regressor",
            "tree_structure": {
                "n_nodes": tree.node_count,
                "n_leaves": tree.n_leaves,
                "max_depth": tree.max_depth,
                "n_features_used": len(np.unique(tree.feature[tree.feature >= 0]))
            },
            "hyperparameters": {
                "criterion": self.criterion,
                "splitter": self.splitter,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "ccp_alpha": self.ccp_alpha
            },
            "interpretability": {
                "highly_interpretable": True,
                "rule_based": True,
                "feature_interactions": True,
                "no_preprocessing_needed": True
            },
            "model_complexity": self.tree_analysis_.get('tree_complexity', {}),
            "performance_analysis": self.cross_validation_analysis_.get('cv_analysis', {})
        }
    
    def get_tree_analysis(self) -> Dict[str, Any]:
        """Get comprehensive tree analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        return {
            "feature_importance_analysis": self.feature_importance_analysis_,
            "tree_analysis": self.tree_analysis_,
            "decision_path_analysis": self.decision_path_analysis_,
            "pruning_analysis": self.pruning_analysis_,
            "overfitting_analysis": self.overfitting_analysis_,
            "complexity_analysis": self.complexity_analysis_,
            "stability_analysis": self.stability_analysis_,
            "cross_validation_analysis": self.cross_validation_analysis_
        }
    
    def plot_tree_analysis(self, figsize=(16, 12)):
        """Plot comprehensive tree analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to plot tree analysis")
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.ravel()
        
        # Plot 1: Tree structure visualization (simplified)
        ax1 = axes[0]
        try:
            if self.model_.tree_.n_leaves <= 20:  # Only for small trees
                from sklearn.tree import plot_tree
                plot_tree(self.model_, ax=ax1, max_depth=3, feature_names=self.feature_names_[:5], 
                         filled=True, rounded=True, fontsize=8)
                ax1.set_title('Tree Structure (Simplified)')
            else:
                ax1.text(0.5, 0.5, f'Tree too complex\nfor visualization\n({self.model_.tree_.n_leaves} leaves)', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=12)
                ax1.set_title('Tree Structure')
                ax1.set_xticks([])
                ax1.set_yticks([])
        except:
            ax1.text(0.5, 0.5, 'Tree visualization\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Tree Structure')
        
        # Plot 2: Feature importance
        ax2 = axes[1]
        if 'builtin_importance' in self.feature_importance_analysis_:
            importance = self.feature_importance_analysis_['builtin_importance']
            feature_names = self.feature_names_[:10]  # Top 10
            importance = importance[:10]
            
            y_pos = np.arange(len(feature_names))
            ax2.barh(y_pos, importance, alpha=0.7, color='skyblue')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([name[:15] for name in feature_names], fontsize=8)
            ax2.set_xlabel('Importance')
            ax2.set_title('Feature Importance (Gini)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Feature Importance')
        
        # Plot 3: Tree complexity metrics
        ax3 = axes[2]
        if 'tree_complexity' in self.tree_analysis_:
            complexity = self.tree_analysis_['tree_complexity']
            
            metrics = ['Nodes', 'Leaves', 'Max Depth', 'Avg Depth']
            values = [
                complexity.get('n_nodes', 0),
                complexity.get('n_leaves', 0),
                complexity.get('max_depth', 0),
                complexity.get('average_depth', 0)
            ]
            
            bars = ax3.bar(metrics, values, alpha=0.7, color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'])
            ax3.set_ylabel('Count/Value')
            ax3.set_title('Tree Complexity Metrics')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Complexity metrics\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Tree Complexity')
        
        # Plot 4: Cross-validation performance
        ax4 = axes[3]
        if 'cv_analysis' in self.cross_validation_analysis_:
            cv_analysis = self.cross_validation_analysis_['cv_analysis']
            cv_scores = cv_analysis['cv_r2_scores']
            
            ax4.boxplot([cv_scores], labels=['CV R²'])
            ax4.scatter([1] * len(cv_scores), cv_scores, alpha=0.7, color='red', s=30)
            ax4.axhline(y=cv_analysis['train_r2'], color='blue', linestyle='--', 
                       label=f'Train R²: {cv_analysis["train_r2"]:.3f}')
            ax4.set_ylabel('R² Score')
            ax4.set_title('Cross-Validation Performance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'CV analysis\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cross-Validation')
        
        # Plot 5: Overfitting analysis
        ax5 = axes[4]
        if 'performance_gaps' in self.overfitting_analysis_:
            gaps = self.overfitting_analysis_['performance_gaps']
            
            metrics = ['R² Gap', 'Relative MSE\nIncrease']
            values = [gaps.get('r2_gap', 0), gaps.get('relative_mse_increase', 0)]
            colors = ['orange' if v > 0.1 else 'green' for v in values]
            
            bars = ax5.bar(metrics, values, color=colors, alpha=0.7)
            ax5.set_ylabel('Gap/Ratio')
            ax5.set_title('Overfitting Analysis')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add threshold lines
            ax5.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'Overfitting analysis\nnot available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Overfitting Analysis')
        
        # Plot 6: Pruning analysis
        ax6 = axes[5]
        if 'tree_performance' in self.pruning_analysis_:
            perf_data = self.pruning_analysis_['tree_performance']
            
            if perf_data:
                alphas = [p['ccp_alpha'] for p in perf_data]
                r2_scores = [p['r2'] for p in perf_data]
                
                ax6.semilogx(alphas, r2_scores, 'o-', alpha=0.7, color='purple')
                ax6.set_xlabel('CCP Alpha')
                ax6.set_ylabel('R² Score')
                ax6.set_title('Pruning Analysis')
                ax6.grid(True, alpha=0.3)
                
                # Mark current alpha
                current_alpha = self.ccp_alpha
                ax6.axvline(x=max(current_alpha, min(alphas)), color='red', linestyle='--', 
                           label=f'Current α: {current_alpha}')
                ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'Pruning analysis\nnot available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Pruning Analysis')
        
        # Plot 7: Stability analysis
        ax7 = axes[6]
        if 'individual_performances' in self.stability_analysis_:
            performances = self.stability_analysis_['individual_performances']
            
            r2_scores = [p['r2'] for p in performances]
            models = range(1, len(r2_scores) + 1)
            
            ax7.plot(models, r2_scores, 'o-', alpha=0.7, color='brown')
            ax7.fill_between(models, r2_scores, alpha=0.3, color='brown')
            ax7.set_xlabel('Model (Random Seed)')
            ax7.set_ylabel('R² Score')
            ax7.set_title('Model Stability')
            ax7.grid(True, alpha=0.3)
            
            # Add mean line
            mean_r2 = np.mean(r2_scores)
            ax7.axhline(y=mean_r2, color='red', linestyle='--', 
                       label=f'Mean: {mean_r2:.3f}')
            ax7.legend()
        else:
            ax7.text(0.5, 0.5, 'Stability analysis\nnot available', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Stability Analysis')
        
        # Plot 8: Decision path lengths
        ax8 = axes[7]
        if 'path_analysis' in self.decision_path_analysis_:
            path_analysis = self.decision_path_analysis_['path_analysis']
            
            # Create histogram of path lengths (simulated)
            mean_length = path_analysis.get('mean_path_length', 5)
            std_length = path_analysis.get('std_path_length', 2)
            
            # Simulate path length distribution
            path_lengths = np.random.normal(mean_length, std_length, 100)
            path_lengths = np.clip(path_lengths, 1, None)
            
            ax8.hist(path_lengths, bins=15, alpha=0.7, color='teal', edgecolor='black')
            ax8.axvline(x=mean_length, color='red', linestyle='--', 
                       label=f'Mean: {mean_length:.1f}')
            ax8.set_xlabel('Decision Path Length')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Decision Path Analysis')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Path analysis\nnot available', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Decision Paths')
        
        # Plot 9: Model summary
        ax9 = axes[8]
        try:
            # Create text summary
            tree = self.model_.tree_
            summary_text = f"""
Tree Summary:
• Nodes: {tree.node_count}
• Leaves: {tree.n_leaves}
• Max Depth: {tree.max_depth}
• Features Used: {len(np.unique(tree.feature[tree.feature >= 0]))}

Complexity: {self.tree_analysis_.get('tree_interpretation', {}).get('complexity_assessment', 'Unknown')}

Overfitting Risk: {self.tree_analysis_.get('tree_interpretation', {}).get('overfitting_risk', 'Unknown')}
"""
            
            ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            ax9.set_title('Model Summary')
            ax9.set_xticks([])
            ax9.set_yticks([])
        except:
            ax9.text(0.5, 0.5, 'Summary not\navailable', 
                    ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Model Summary')
        
        plt.tight_layout()
        return fig
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        tree = self.model_.tree_
        
        info = {
            "algorithm": "Decision Tree Regressor",
            "type": "Non-linear, rule-based regression with high interpretability",
            "training_completed": True,
            "tree_characteristics": {
                "non_parametric": True,
                "highly_interpretable": True,
                "handles_non_linearity": True,
                "automatic_feature_selection": True,
                "no_preprocessing_required": True,
                "robust_to_outliers": True
            },
            "tree_structure": {
                "n_nodes": tree.node_count,
                "n_leaves": tree.n_leaves,
                "max_depth": tree.max_depth,
                "n_features_used": len(np.unique(tree.feature[tree.feature >= 0])),
                "total_features": self.n_features_in_
            },
            "model_configuration": {
                "criterion": self.criterion,
                "splitter": self.splitter,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "ccp_alpha": self.ccp_alpha
            }
        }
        
        # Add analysis information if available
        if self.tree_analysis_:
            info["complexity_analysis"] = {
                "complexity_assessment": self.tree_analysis_.get('tree_interpretation', {}).get('complexity_assessment'),
                "interpretability_score": self.tree_analysis_.get('tree_interpretation', {}).get('interpretability_score'),
                "overfitting_risk": self.tree_analysis_.get('tree_interpretation', {}).get('overfitting_risk')
            }
        
        if self.cross_validation_analysis_:
            info["performance_analysis"] = {
                "cv_r2_mean": self.cross_validation_analysis_.get('cv_analysis', {}).get('cv_r2_mean'),
                "generalization_gap": self.cross_validation_analysis_.get('cv_analysis', {}).get('generalization_gap'),
                "performance_assessment": self.cross_validation_analysis_.get('performance_assessment', {}).get('overall_assessment')
            }
        
        return info

    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Decision Tree Regressor-specific metrics based on the fitted model's
        internal analyses.

        Note: Most metrics are derived from analyses performed on the training data
        or internal cross-validation during the fit method.
        The y_true, y_pred, y_proba parameters (typically for test set evaluation)
        are not directly used for these internal tree-specific metrics.

        Args:
            y_true: Ground truth target values from a test set.
            y_pred: Predicted target values on a test set.
            y_proba: Predicted probabilities (not typically used by DecisionTreeRegressor).

        Returns:
            A dictionary of Decision Tree Regressor-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted"
            return metrics

        metrics['tree_criterion_used'] = self.model_.criterion

        # --- Metrics from Tree Structure Analysis ---
        if hasattr(self, 'tree_analysis_') and self.tree_analysis_ and 'error' not in self.tree_analysis_:
            if 'tree_complexity' in self.tree_analysis_:
                tc = self.tree_analysis_['tree_complexity']
                metrics['tree_n_nodes'] = tc.get('n_nodes')
                metrics['tree_n_leaves'] = tc.get('n_leaves')
                metrics['tree_max_depth_actual'] = tc.get('max_depth')
                metrics['tree_average_depth'] = tc.get('average_depth')
                metrics['tree_balance_ratio'] = tc.get('balance_ratio')
                metrics['tree_average_leaf_purity'] = tc.get('leaf_purity') # 1 - weighted impurity

            if 'tree_interpretation' in self.tree_analysis_:
                ti = self.tree_analysis_['tree_interpretation']
                metrics['tree_interpretability_score'] = ti.get('interpretability_score')

        # --- Metrics from Feature Importance Analysis ---
        if hasattr(self, 'feature_importance_analysis_') and self.feature_importance_analysis_ and \
           'error' not in self.feature_importance_analysis_:
            fia = self.feature_importance_analysis_
            if 'builtin_importance' in fia and fia['builtin_importance'] is not None:
                metrics['num_features_used_in_tree'] = int(np.sum(np.array(fia['builtin_importance']) > 0))
            if 'importance_statistics' in fia:
                stats = fia['importance_statistics']
                metrics['feature_importance_gini_concentration'] = stats.get('gini_coefficient')
                metrics['mean_builtin_feature_importance'] = stats.get('mean_importance')
            if 'permutation_importance' in fia and fia['permutation_importance'] is not None:
                try:
                    metrics['mean_permutation_importance'] = float(np.mean(fia['permutation_importance']))
                except TypeError: # handles if it's None or not array-like
                    metrics['mean_permutation_importance'] = None


        # --- Metrics from Complexity Analysis ---
        if hasattr(self, 'complexity_analysis_') and self.complexity_analysis_ and \
           'error' not in self.complexity_analysis_:
            ca = self.complexity_analysis_
            metrics['model_complexity_score'] = ca.get('complexity_score')
            if 'structure_complexity' in ca:
                metrics['average_samples_per_leaf'] = ca['structure_complexity'].get('average_samples_per_leaf')

        # --- Metrics from Pruning Analysis ---
        metrics['ccp_alpha_applied'] = self.ccp_alpha
        if hasattr(self, 'pruning_analysis_') and self.pruning_analysis_ and \
           'error' not in self.pruning_analysis_:
            metrics['optimal_ccp_alpha_suggested'] = self.pruning_analysis_.get('optimal_alpha')

        # --- Metrics from Stability Analysis ---
        if hasattr(self, 'stability_analysis_') and self.stability_analysis_ and \
           'error' not in self.stability_analysis_:
            if 'stability_assessment' in self.stability_analysis_:
                sa = self.stability_analysis_['stability_assessment']
                metrics['stability_r2_coeff_variation'] = sa.get('r2_coefficient_of_variation')
                metrics['stability_structure_coeff_variation'] = sa.get('structure_coefficient_of_variation')

        # --- Metrics from Cross-Validation Analysis ---
        if hasattr(self, 'cross_validation_analysis_') and self.cross_validation_analysis_ and \
           'error' not in self.cross_validation_analysis_:
            if 'cv_analysis' in self.cross_validation_analysis_:
                cva = self.cross_validation_analysis_['cv_analysis']
                metrics['cv_r2_mean_internal'] = cva.get('cv_r2_mean')
                metrics['cv_r2_std_internal'] = cva.get('cv_r2_std')
                metrics['cv_generalization_gap_r2_internal'] = cva.get('generalization_gap')
            if 'performance_assessment' in self.cross_validation_analysis_:
                 metrics['cv_reliability_internal'] = self.cross_validation_analysis_['performance_assessment'].get('reliability')


        # --- Metrics from Overfitting Analysis (internal train/val split) ---
        if hasattr(self, 'overfitting_analysis_') and self.overfitting_analysis_ and \
           'error' not in self.overfitting_analysis_:
            if 'performance_gaps' in self.overfitting_analysis_:
                gaps = self.overfitting_analysis_['performance_gaps']
                metrics['overfitting_internal_r2_gap'] = gaps.get('r2_gap')
                metrics['overfitting_internal_relative_mse_increase'] = gaps.get('relative_mse_increase')
            if 'overfitting_assessment' in self.overfitting_analysis_:
                metrics['overfitting_internal_assessment_level'] = self.overfitting_analysis_['overfitting_assessment'].get('level')

        # Remove None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if v is not None}

        return metrics


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return DecisionTreeRegressorPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Decision Tree Regressor Plugin
    """
    print("Testing Decision Tree Regressor Plugin...")
    
    try:
        # Create sample data with non-linear relationships
        np.random.seed(42)
        
        # Generate synthetic non-linear regression data
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=500,
            n_features=10,
            n_informative=6,
            noise=0.1,
            random_state=42
        )
        
        # Add some non-linear transformations
        y = y + 0.1 * X[:, 0] * X[:, 1] + 0.05 * X[:, 2]**2
        
        print(f"\n📊 Test Dataset:")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"Target variance: {np.var(y):.3f}")
        print(f"Non-linear relationships added")
        
        # Test Decision Tree regression
        print(f"\n🧪 Testing Decision Tree Regression...")
        
        # Create DataFrame for proper feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        plugin = DecisionTreeRegressorPlugin(
            criterion='squared_error',
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=5,
            ccp_alpha=0.001,
            compute_feature_importance=True,
            compute_permutation_importance=True,
            tree_interpretation_analysis=True,
            overfitting_analysis=True,
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
            
            # Evaluate
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            print(f"\n📊 Decision Tree Results:")
            print(f"R²: {r2:.4f}")
            print(f"MSE: {mse:.4f}")
            
            # Get model parameters
            model_params = plugin.get_model_params()
            tree_structure = model_params['tree_structure']
            print(f"\nTree Structure:")
            print(f"Nodes: {tree_structure['n_nodes']}")
            print(f"Leaves: {tree_structure['n_leaves']}")
            print(f"Max Depth: {tree_structure['max_depth']}")
            print(f"Features Used: {tree_structure['n_features_used']}/{tree_structure.get('total_features', 'N/A')}")
            
            # Test feature importance
            importance = plugin.get_feature_importance()
            if importance and 'top_features' in importance:
                print(f"\n🎯 Top Features:")
                for i, (name, gini_imp, perm_imp) in enumerate(importance['top_features'][:5]):
                    perm_str = f", Perm: {perm_imp:.4f}" if perm_imp is not None else ""
                    print(f"  {i+1}. {name}: Gini: {gini_imp:.4f}{perm_str}")
            
            # Get comprehensive analysis
            analysis = plugin.get_tree_analysis()
            
            # Check complexity
            if 'tree_analysis' in analysis and 'tree_interpretation' in analysis['tree_analysis']:
                interp = analysis['tree_analysis']['tree_interpretation']
                print(f"\nTree Analysis:")
                print(f"Complexity: {interp.get('complexity_assessment', 'Unknown')}")
                print(f"Interpretability Score: {interp.get('interpretability_score', 'Unknown'):.3f}")
                print(f"Overfitting Risk: {interp.get('overfitting_risk', 'Unknown')}")
            
            # Check cross-validation performance
            if 'cross_validation_analysis' in analysis and 'performance_assessment' in analysis['cross_validation_analysis']:
                cv_perf = analysis['cross_validation_analysis']['performance_assessment']
                print(f"CV Performance: {cv_perf.get('overall_assessment', 'Unknown')}")
                print(f"CV R² Mean: {cv_perf.get('cv_mean_r2', 'Unknown'):.4f}")
                print(f"Generalization Gap: {cv_perf.get('generalization_gap', 'Unknown'):.4f}")
        
        print("\n✅ Decision Tree Regressor Plugin test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing Decision Tree Plugin: {str(e)}")
        import traceback
        traceback.print_exc()