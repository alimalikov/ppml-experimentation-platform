import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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

class ExtraTreesClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Extra Trees Classifier Plugin - Faster variant of Random Forest
    
    Extra Trees (Extremely Randomized Trees) builds an ensemble of unpruned 
    decision trees using random splits instead of best splits, making it 
    faster than Random Forest while maintaining similar performance.
    """
    
    def __init__(self, 
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='sqrt',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=42,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None):
        """
        Initialize Extra Trees Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        n_estimators : int, default=100
            The number of trees in the forest
        criterion : {'gini', 'entropy', 'log_loss'}, default='gini'
            The function to measure the quality of a split
        max_depth : int, default=None
            The maximum depth of the tree
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node
        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node
        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights
        max_features : {'sqrt', 'log2', None}, int or float, default='sqrt'
            The number of features to consider when looking for the best split
        max_leaf_nodes : int, default=None
            Grow trees with max_leaf_nodes in best-first fashion
        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
        bootstrap : bool, default=False
            Whether bootstrap samples are used when building trees (default False for Extra Trees)
        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate the generalization score
        n_jobs : int, default=None
            The number of jobs to run in parallel for both fit and predict
        random_state : int, default=42
            Controls randomness of the estimator
        verbose : int, default=0
            Controls the verbosity when fitting and predicting
        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit
        class_weight : {'balanced', 'balanced_subsample'}, dict or list of dicts, default=None
            Weights associated with classes
        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning
        max_samples : int or float, default=None
            If bootstrap is True, the number of samples to draw from X
        """
        super().__init__()
        
        # Algorithm parameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        
        # Plugin metadata
        self._name = "Extra Trees"
        self._description = "Extremely Randomized Trees - Faster variant of Random Forest using random splits instead of best splits."
        self._category = "Tree-Based Models"
        self._algorithm_type = "Ensemble Classifier"
        self._paper_reference = "Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine learning, 63(1), 3-42."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 20
        self._handles_missing_values = False
        self._requires_scaling = False
        self._supports_sparse = True
        self._is_linear = False
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = True
        self._ensemble_method = True
        self._supports_oob_score = True
        self._parallel_training = True
        self._extremely_randomized = True
        self._faster_than_rf = True
        
        # Internal attributes
        self.model_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        
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
            "ensemble_type": "Extremely Randomized Trees",
            "key_difference_from_rf": {
                "split_selection": "Random splits vs. best splits",
                "speed": "Faster training due to random splits",
                "randomness": "Higher randomness in split selection",
                "bootstrap": "Default bootstrap=False (uses all data)",
                "bias_variance": "Higher bias, lower variance than Random Forest"
            },
            "strengths": [
                "Faster training than Random Forest",
                "Reduces overfitting through extreme randomization",
                "Excellent performance on many datasets",
                "Handles mixed data types naturally",
                "Provides feature importance rankings",
                "Robust to outliers and noise",
                "Parallel training capability",
                "Lower computational complexity",
                "Good generalization due to high randomness",
                "Requires minimal hyperparameter tuning"
            ],
            "weaknesses": [
                "May have slightly higher bias than Random Forest",
                "Less interpretable than single decision trees",
                "Memory intensive with many trees",
                "Can be slower than simpler models for small datasets",
                "May not capture optimal splits as well as Random Forest",
                "Performance depends on ensemble size"
            ],
            "use_cases": [
                "Large datasets where training speed matters",
                "High-dimensional feature spaces",
                "When Random Forest overfits",
                "Feature importance analysis",
                "Structured/tabular data problems",
                "Baseline ensemble model",
                "Computer vision with tabular features",
                "Bioinformatics and genomics",
                "Finance and risk assessment",
                "Any domain requiring fast ensemble training"
            ],
            "algorithmic_details": {
                "randomization_sources": [
                    "Random subset of training samples",
                    "Random subset of features at each split",
                    "Random split points instead of optimal splits",
                    "No pruning of individual trees"
                ],
                "split_selection": "Randomly select K features, then randomly select threshold",
                "combination_method": "Majority voting for classification",
                "variance_reduction": "Extreme randomization reduces variance"
            },
            "complexity": {
                "training": "O(n × log(n) × m × k) - faster than RF due to random splits",
                "prediction": "O(log(n) × k)",
                "space": "O(n × k)"
            },
            "comparison_with_rf": {
                "speed": "Faster training (random vs. best splits)",
                "accuracy": "Often similar, sometimes slightly lower",
                "overfitting": "Less prone to overfitting",
                "randomness": "Higher randomness level",
                "use_case": "When speed and generalization are priorities"
            },
            "parameters_guide": {
                "n_estimators": "More trees = better performance but slower (50-500)",
                "max_depth": "Controls individual tree complexity (None or 10-30)",
                "max_features": "sqrt: good default, higher values for more randomness",
                "bootstrap": "False: default, True: adds another randomization layer",
                "min_samples_split": "Higher values = more regularization"
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Extra Trees Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
            
        Returns:
        --------
        self : object
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        
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
        
        # Create and configure the Extra Trees model
        self.model_ = ExtraTreesClassifier(
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
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples
        )
        
        # Train the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if sample_weight is not None:
                self.model_.fit(X, y_encoded, sample_weight=sample_weight)
            else:
                self.model_.fit(X, y_encoded)
        
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
        X = check_array(X, accept_sparse=True, dtype=None)
        
        # Make predictions
        y_pred_encoded = self.model_.predict(X)
        
        # Decode labels back to original format
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
        X = check_array(X, accept_sparse=True, dtype=None)
        
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance based on impurity decrease
        
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores
        """
        if not self.is_fitted_:
            return None
            
        return self.model_.feature_importances_
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available
        
        Returns:
        --------
        oob_score : float or None
            Out-of-bag score
        """
        if not self.is_fitted_ or not self.oob_score:
            return None
            
        return getattr(self.model_, 'oob_score_', None)
    
    def get_individual_tree_predictions(self, X, tree_idx=0):
        """
        Get predictions from an individual tree
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        tree_idx : int, default=0
            Index of the tree to get predictions from
            
        Returns:
        --------
        predictions : array
            Predictions from the specified tree
        """
        if not self.is_fitted_:
            return None
        
        if tree_idx >= len(self.model_.estimators_):
            return None
        
        X = check_array(X, accept_sparse=True, dtype=None)
        tree_pred = self.model_.estimators_[tree_idx].predict(X)
        
        # Decode labels back to original format
        tree_pred_decoded = self.label_encoder_.inverse_transform(tree_pred)
        
        return tree_pred_decoded
    
    def analyze_tree_diversity(self) -> Dict[str, Any]:
        """
        Analyze diversity among trees in the forest
        
        Returns:
        --------
        diversity_info : dict
            Information about tree diversity
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        # Get tree depths
        tree_depths = [tree.tree_.max_depth for tree in self.model_.estimators_]
        
        # Get number of nodes per tree
        tree_nodes = [tree.tree_.node_count for tree in self.model_.estimators_]
        
        # Get number of leaves per tree
        tree_leaves = [tree.tree_.n_leaves for tree in self.model_.estimators_]
        
        diversity_info = {
            "n_trees": len(self.model_.estimators_),
            "randomization_level": "Extremely High (random splits)",
            "depth_stats": {
                "mean": np.mean(tree_depths),
                "std": np.std(tree_depths),
                "min": np.min(tree_depths),
                "max": np.max(tree_depths)
            },
            "nodes_stats": {
                "mean": np.mean(tree_nodes),
                "std": np.std(tree_nodes),
                "min": np.min(tree_nodes),
                "max": np.max(tree_nodes)
            },
            "leaves_stats": {
                "mean": np.mean(tree_leaves),
                "std": np.std(tree_leaves),
                "min": np.min(tree_leaves),
                "max": np.max(tree_leaves)
            },
            "diversity_sources": [
                "Random feature subsets at each split",
                "Random split thresholds (not optimal)",
                "Different training samples (if bootstrap=True)",
                "No pruning applied"
            ]
        }
        
        return diversity_info
    
    def get_randomization_analysis(self) -> Dict[str, Any]:
        """
        Analyze the randomization aspects specific to Extra Trees
        
        Returns:
        --------
        randomization_info : dict
            Information about randomization in the model
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "algorithm": "Extra Trees (Extremely Randomized Trees)",
            "randomization_level": "Maximum",
            "key_differences_from_rf": {
                "split_selection": "Random thresholds vs. optimal splits",
                "speed_advantage": "~2-5x faster training than Random Forest",
                "bias_variance_tradeoff": "Higher bias, lower variance"
            },
            "randomization_sources": {
                "feature_sampling": f"Random {self.max_features} features per split",
                "split_thresholds": "Completely random (not optimized)",
                "sample_selection": "Bootstrap sampling" if self.bootstrap else "All samples used",
                "tree_structure": "No pruning (grows full trees)"
            },
            "parameters_affecting_randomness": {
                "max_features": self.max_features,
                "bootstrap": self.bootstrap,
                "random_state": self.random_state,
                "n_estimators": self.n_estimators
            },
            "computational_advantages": [
                "No need to evaluate multiple split points",
                "No need to sort features for optimal splits",
                "Parallel tree construction",
                "Lower memory usage during training"
            ]
        }
        
        return analysis
    
    def plot_feature_importance(self, top_n=20, figsize=(10, 8)):
        """
        Create a feature importance plot
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to display
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Feature importance plot
        """
        if not self.is_fitted_:
            return None
        
        importance = self.get_feature_importance()
        if importance is None:
            return None
        
        # Get top features
        indices = np.argsort(importance)[::-1][:top_n]
        top_features = [self.feature_names_[i] for i in indices]
        top_importance = importance[indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(top_features)), top_importance, 
                      color='forestgreen', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {len(top_features)} Feature Importances - Extra Trees')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_vs_random_forest_comparison(self, figsize=(12, 8)):
        """
        Create a visual comparison between Extra Trees and Random Forest characteristics
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Comparison visualization
        """
        if not self.is_fitted_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Comparison data
        categories = ['Training Speed', 'Randomness Level', 'Variance', 'Bias']
        et_values = [9, 10, 3, 7]  # Extra Trees scores
        rf_values = [6, 7, 5, 5]   # Random Forest scores
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Bar comparison
        ax1.bar(x - width/2, et_values, width, label='Extra Trees', color='forestgreen', alpha=0.8)
        ax1.bar(x + width/2, rf_values, width, label='Random Forest', color='brown', alpha=0.8)
        ax1.set_xlabel('Characteristics')
        ax1.set_ylabel('Score (1-10)')
        ax1.set_title('Extra Trees vs Random Forest Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tree diversity visualization
        diversity_info = self.analyze_tree_diversity()
        if 'depth_stats' in diversity_info:
            ax2.hist([diversity_info['depth_stats']['mean']], bins=1, alpha=0.7, 
                    color='forestgreen', label='Avg Tree Depth')
            ax2.set_xlabel('Tree Depth')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Tree Depth Distribution')
            ax2.legend()
        
        # Randomization sources pie chart
        randomization_sources = ['Random Features', 'Random Splits', 'Sample Selection', 'No Pruning']
        sizes = [25, 35, 20, 20]  # Approximate importance
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        ax3.pie(sizes, labels=randomization_sources, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Sources of Randomization in Extra Trees')
        
        # Algorithm complexity comparison
        algorithms = ['Decision Tree', 'Random Forest', 'Extra Trees']
        training_complexity = [3, 7, 5]  # Relative complexity
        prediction_speed = [9, 7, 8]     # Relative speed
        
        ax4.scatter(training_complexity, prediction_speed, s=[100, 200, 150], 
                   c=['blue', 'brown', 'forestgreen'], alpha=0.7)
        
        for i, alg in enumerate(algorithms):
            ax4.annotate(alg, (training_complexity[i], prediction_speed[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('Training Complexity')
        ax4.set_ylabel('Prediction Speed')
        ax4.set_title('Algorithm Complexity vs Speed')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🌲⚡ Extra Trees Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3 = st.sidebar.tabs(["Core", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Number of estimators
            n_estimators = st.slider(
                "Number of Trees:",
                min_value=10,
                max_value=500,
                value=int(self.n_estimators),
                step=10,
                help="More trees = better performance. Extra Trees often needs fewer than Random Forest",
                key=f"{key_prefix}_n_estimators"
            )
            
            # Max depth
            max_depth_enabled = st.checkbox(
                "Limit Tree Depth",
                value=self.max_depth is not None,
                help="Prevent overfitting by limiting individual tree depth",
                key=f"{key_prefix}_max_depth_enabled"
            )
            
            if max_depth_enabled:
                max_depth = st.slider(
                    "Max Depth:",
                    min_value=1,
                    max_value=30,
                    value=int(self.max_depth) if self.max_depth else 10,
                    help="Maximum depth of individual trees",
                    key=f"{key_prefix}_max_depth"
                )
            else:
                max_depth = None
            
            # Max features
            max_features_option = st.selectbox(
                "Max Features per Split:",
                options=['sqrt', 'log2', 'None', 'custom'],
                index=0 if self.max_features == 'sqrt' else 
                      1 if self.max_features == 'log2' else 
                      2 if self.max_features is None else 3,
                help="sqrt: Good default, log2: More randomness, None: All features",
                key=f"{key_prefix}_max_features_option"
            )
            
            if max_features_option == 'custom':
                max_features_value = st.slider(
                    "Number of Features:",
                    min_value=1,
                    max_value=min(50, self.n_features_in_ if hasattr(self, 'n_features_in_') else 20),
                    value=5,
                    key=f"{key_prefix}_max_features_value"
                )
                max_features = max_features_value
            elif max_features_option == 'None':
                max_features = None
            else:
                max_features = max_features_option
            
            # Min samples split
            min_samples_split = st.slider(
                "Min Samples Split:",
                min_value=2,
                max_value=50,
                value=int(self.min_samples_split),
                help="Minimum samples required to split an internal node",
                key=f"{key_prefix}_min_samples_split"
            )
            
            # Min samples leaf
            min_samples_leaf = st.slider(
                "Min Samples Leaf:",
                min_value=1,
                max_value=20,
                value=int(self.min_samples_leaf),
                help="Minimum samples required to be at a leaf node",
                key=f"{key_prefix}_min_samples_leaf"
            )
        
        with tab2:
            st.markdown("**Advanced Parameters**")
            
            # Criterion
            criterion = st.selectbox(
                "Split Criterion:",
                options=['gini', 'entropy', 'log_loss'],
                index=['gini', 'entropy', 'log_loss'].index(self.criterion),
                help="gini: Fast, entropy: Information gain, log_loss: Logistic loss",
                key=f"{key_prefix}_criterion"
            )
            
            # Bootstrap (default False for Extra Trees)
            bootstrap = st.checkbox(
                "Bootstrap Sampling",
                value=self.bootstrap,
                help="Use bootstrap samples when building trees (default False for Extra Trees)",
                key=f"{key_prefix}_bootstrap"
            )
            
            if bootstrap:
                st.info("💡 Bootstrap=True adds another layer of randomization to Extra Trees")
            else:
                st.info("ℹ️ Extra Trees default: uses all training data (no bootstrap)")
            
            # OOB Score
            oob_score = st.checkbox(
                "Out-of-Bag Score",
                value=self.oob_score,
                help="Estimate generalization error using out-of-bag samples",
                key=f"{key_prefix}_oob_score"
            )
            
            if not bootstrap and oob_score:
                st.warning("⚠️ OOB score requires bootstrap=True")
                oob_score = False
            
            # Class weight
            class_weight_option = st.selectbox(
                "Class Weight:",
                options=['None', 'balanced', 'balanced_subsample'],
                index=0 if self.class_weight is None else 
                      1 if self.class_weight == 'balanced' else 2,
                help="balanced: Adjust weights, balanced_subsample: Per-tree balancing",
                key=f"{key_prefix}_class_weight"
            )
            class_weight = None if class_weight_option == 'None' else class_weight_option
            
            # Max samples
            max_samples_enabled = st.checkbox(
                "Limit Sample Size per Tree",
                value=self.max_samples is not None,
                help="Limit number of samples used per tree",
                key=f"{key_prefix}_max_samples_enabled"
            )
            
            if max_samples_enabled:
                max_samples = st.slider(
                    "Max Samples Ratio:",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(self.max_samples) if self.max_samples else 1.0,
                    step=0.1,
                    help="Fraction of samples to use per tree",
                    key=f"{key_prefix}_max_samples"
                )
            else:
                max_samples = None
            
            # Min impurity decrease
            min_impurity_decrease = st.number_input(
                "Min Impurity Decrease:",
                value=float(self.min_impurity_decrease),
                min_value=0.0,
                max_value=0.1,
                step=0.001,
                format="%.4f",
                help="Minimum impurity decrease required for split",
                key=f"{key_prefix}_min_impurity_decrease"
            )
            
            # Random state
            random_state = st.number_input(
                "Random State:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
        
        with tab3:
            st.markdown("**Algorithm Information**")
            st.info("""
            **Extra Trees** excels at:
            
            • Faster training than Random Forest
            
            • Extreme randomization prevents overfitting
            
            • Excellent generalization ability
            
            • Handling high-dimensional data
            
            • Feature importance analysis
            
            **Key Difference from Random Forest:**
            
            • Uses **random splits** instead of best splits
            
            • **2-5x faster** training time
            
            • **Higher bias, lower variance**
            
            • Default **no bootstrap** sampling
            """)
            
            # Randomization showcase
            if st.button("🎲 Randomization Explained", key=f"{key_prefix}_randomization"):
                st.markdown("""
                **Extra Trees Randomization:**
                
                1. **Random Feature Selection**: At each split, randomly select K features
                2. **Random Split Points**: Instead of finding the best threshold, randomly select one
                3. **No Optimization**: No need to evaluate all possible splits
                4. **Result**: Much faster training with similar accuracy
                
                **vs Random Forest:**
                - RF: Find best split among random features
                - ET: Use random split among random features
                """)
            
            # Performance comparison
            if st.button("⚡ Speed vs Accuracy", key=f"{key_prefix}_speed_comparison"):
                st.markdown("""
                **Training Speed Comparison:**
                - Extra Trees: ⭐⭐⭐⭐⭐ (Fastest ensemble)
                - Random Forest: ⭐⭐⭐ (Moderate)
                - Decision Tree: ⭐⭐⭐⭐⭐ (Fastest single)
                
                **Accuracy Comparison:**
                - Extra Trees: ⭐⭐⭐⭐ (Excellent)
                - Random Forest: ⭐⭐⭐⭐⭐ (Excellent)
                - Decision Tree: ⭐⭐⭐ (Good but overfits)
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": 0.0,
            "max_features": max_features,
            "max_leaf_nodes": None,
            "min_impurity_decrease": min_impurity_decrease,
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "n_jobs": -1,  # Use all cores for better performance
            "random_state": random_state,
            "verbose": 0,
            "warm_start": False,
            "class_weight": class_weight,
            "ccp_alpha": 0.0,
            "max_samples": max_samples
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return ExtraTreesClassifierPlugin(
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
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            ccp_alpha=hyperparameters.get("ccp_alpha", self.ccp_alpha),
            max_samples=hyperparameters.get("max_samples", self.max_samples)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        Extra Trees requires minimal preprocessing:
        1. Can handle mixed data types
        2. No scaling required
        3. Robust to outliers
        4. Even more robust than Random Forest due to extreme randomization
        """
        return X, y
    
    def is_compatible_with_data(self, df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
        """
        Check if algorithm is compatible with the data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target_column : str
            Name of target column
            
        Returns:
        --------
        is_compatible : bool
            Whether the algorithm can handle this data
        message : str
            Detailed compatibility message
        """
        try:
            # Check if target column exists
            if target_column not in df.columns:
                return False, f"Target column '{target_column}' not found in dataset"
            
            # Check dataset size
            n_samples, n_features = df.shape
            if n_samples < self._min_samples_required:
                return False, f"Minimum {self._min_samples_required} samples required, got {n_samples}"
            
            # Check target variable type
            target_values = df[target_column].unique()
            n_classes = len(target_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). Extra Trees works better with fewer classes."
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            has_missing = missing_values > 0
            
            # Check feature types
            feature_columns = [col for col in df.columns if col != target_column]
            numeric_features = []
            categorical_features = []
            
            for col in feature_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_features.append(col)
                else:
                    categorical_features.append(col)
            
            # Extra Trees specific advantages
            advantages = []
            considerations = []
            
            # Speed advantage
            if n_samples >= 1000 or n_features >= 20:
                advantages.append(f"Large dataset ({n_samples} samples, {n_features} features) - Extra Trees will be significantly faster than Random Forest")
            
            # High-dimensional advantage
            if n_features >= 50:
                advantages.append(f"High-dimensional data ({n_features} features) - extreme randomization handles this well")
            
            # Mixed data types
            if len(categorical_features) > 0 and len(numeric_features) > 0:
                advantages.append(f"Excellent for mixed data types ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
            
            # Overfitting resistance
            if n_features > n_samples / 10:
                advantages.append("High feature-to-sample ratio - Extra Trees' extreme randomization prevents overfitting")
            
            # Dataset size considerations
            if n_samples >= 10000:
                advantages.append(f"Large dataset ({n_samples} samples) - parallel training will be very fast")
            elif n_samples < 100:
                considerations.append(f"Small dataset ({n_samples} samples) - may not fully benefit from ensemble, consider fewer trees")
            
            # Feature count optimization
            if n_features <= 5:
                considerations.append(f"Few features ({n_features}) - may not benefit much from feature randomness")
            elif n_features >= 100:
                advantages.append(f"Many features ({n_features}) - random feature selection will be very effective")
            
            # No scaling needed
            if len(numeric_features) > 0:
                ranges = []
                for col in numeric_features[:5]:
                    try:
                        col_range = df[col].max() - df[col].min()
                        if col_range > 0:
                            ranges.append(col_range)
                    except:
                        pass
                
                if len(ranges) > 1 and max(ranges) / min(ranges) > 100:
                    advantages.append("No scaling required (robust to different feature scales)")
            
            # Class balance
            if target_column in df.columns:
                class_counts = df[target_column].value_counts()
                min_class_size = class_counts.min()
                max_class_size = class_counts.max()
                
                if max_class_size / min_class_size > 10:
                    considerations.append("Imbalanced classes detected - consider class_weight='balanced'")
                else:
                    advantages.append("Well-balanced classes for ensemble learning")
            
            # Missing values handling
            if has_missing:
                if len(categorical_features) > 0:
                    return False, f"Missing values detected ({missing_values} total). Please handle missing values first - Extra Trees requires complete data."
                else:
                    considerations.append(f"Missing values detected ({missing_values}) - will need preprocessing")
            
            # Categorical encoding needed
            if len(categorical_features) > 0:
                return False, f"Categorical features detected: {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}. Please encode categorical variables first."
            
            # Performance optimization suggestions
            if n_samples * n_features > 100000:  # Large dataset
                advantages.append("Large dataset - Extra Trees' speed advantage will be significant")
            
            # When to prefer Extra Trees over Random Forest
            speed_preference_factors = []
            if n_samples > 10000:
                speed_preference_factors.append("large sample size")
            if n_features > 50:
                speed_preference_factors.append("many features")
            
            if speed_preference_factors:
                advantages.append(f"Ideal for Extra Trees due to: {', '.join(speed_preference_factors)}")
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if advantages:
                message_parts.append("🚀 Extra Trees advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'max_samples': self.max_samples
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        # Basic info
        info = {
            "status": "Fitted",
            "algorithm": "Extra Trees (Extremely Randomized Trees)",
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": list(self.classes_),
            "feature_names": self.feature_names_,
            "n_estimators": self.n_estimators
        }
        
        # OOB Score if available
        oob_score = self.get_oob_score()
        if oob_score is not None:
            info["oob_score"] = float(oob_score)
        
        # Tree diversity analysis
        diversity_info = self.analyze_tree_diversity()
        info["tree_diversity"] = diversity_info
        
        # Randomization analysis
        randomization_info = self.get_randomization_analysis()
        info["randomization_analysis"] = randomization_info
        
        # Feature importance
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            # Get top 5 most important features
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            top_features = {
                "top_features": [
                    {
                        "feature": self.feature_names_[idx],
                        "importance": float(feature_importance[idx])
                    }
                    for idx in top_features_idx
                ],
                "importance_concentration": {
                    "top_5_sum": float(np.sum(feature_importance[top_features_idx])),
                    "entropy": float(-np.sum(feature_importance * np.log(feature_importance + 1e-10)))
                }
            }
            info.update(top_features)
        
        return info

    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Extra Trees Classifier-specific metrics based on the fitted model's
        internal analyses and characteristics.

        Note: Most metrics are derived from the model structure and training process.
        The y_true, y_pred, y_proba parameters (typically for test set evaluation)
        are not directly used for these internal ensemble-specific metrics,
        though y_proba could be used for metrics like log_loss if desired externally.

        Args:
            y_true: Ground truth target values from a test set.
            y_pred: Predicted target values on a test set.
            y_proba: Predicted probabilities on a test set.

        Returns:
            A dictionary of Extra Trees Classifier-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted"
            return metrics

        # --- OOB Score ---
        if self.oob_score and hasattr(self.model_, 'oob_score_'):
            metrics['oob_score_value'] = float(self.model_.oob_score_)

        # --- Tree Diversity Metrics ---
        if hasattr(self.model_, 'estimators_') and self.model_.estimators_:
            diversity_info = self.analyze_tree_diversity() # This method is already in the class
            if 'depth_stats' in diversity_info:
                metrics['mean_tree_depth'] = float(diversity_info['depth_stats'].get('mean', np.nan))
                metrics['std_tree_depth'] = float(diversity_info['depth_stats'].get('std', np.nan))
            if 'nodes_stats' in diversity_info:
                metrics['mean_tree_nodes'] = float(diversity_info['nodes_stats'].get('mean', np.nan))
                metrics['std_tree_nodes'] = float(diversity_info['nodes_stats'].get('std', np.nan))
            if 'leaves_stats' in diversity_info:
                metrics['mean_tree_leaves'] = float(diversity_info['leaves_stats'].get('mean', np.nan))
                metrics['std_tree_leaves'] = float(diversity_info['leaves_stats'].get('std', np.nan))

        # --- Feature Importance Metrics ---
        importances = self.get_feature_importance() # This method is already in the class
        if importances is not None and len(importances) > 0:
            metrics['num_features_model_used'] = int(np.sum(importances > 0))
            metrics['mean_feature_importance'] = float(np.mean(importances))
            
            # Calculate Gini coefficient for feature importance concentration
            sorted_importances = np.sort(importances)
            n = len(sorted_importances)
            if n > 1 and np.sum(sorted_importances) > 0:
                index = np.arange(1, n + 1)
                gini_coeff = (np.sum((2 * index - n - 1) * sorted_importances)) / (n * np.sum(sorted_importances))
                metrics['feature_importance_gini_concentration'] = float(gini_coeff)
            else:
                metrics['feature_importance_gini_concentration'] = 0.0 if n <=1 else np.nan


        # --- Model Configuration Metrics ---
        metrics['num_estimators_actual'] = self.model_.n_estimators
        
        # Resolve max_features to an integer if it's a string
        if isinstance(self.model_.max_features, str):
            if self.model_.max_features == 'sqrt':
                metrics['max_features_resolved_per_split'] = int(np.sqrt(self.model_.n_features_in_))
            elif self.model_.max_features == 'log2':
                metrics['max_features_resolved_per_split'] = int(np.log2(self.model_.n_features_in_))
            else: # Should not happen with current sklearn options if not None
                 metrics['max_features_resolved_per_split'] = self.model_.n_features_in_
        elif self.model_.max_features is None:
            metrics['max_features_resolved_per_split'] = self.model_.n_features_in_
        else: # It's an int or float
            metrics['max_features_resolved_per_split'] = int(self.model_.max_features * self.model_.n_features_in_
                                                             if isinstance(self.model_.max_features, float)
                                                             else self.model_.max_features)
        
        metrics['criterion_used'] = self.model_.criterion

        # Remove NaN or None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if pd.notna(v)}

        return metrics

def get_plugin():
    """Factory function to get plugin instance"""
    return ExtraTreesClassifierPlugin()
