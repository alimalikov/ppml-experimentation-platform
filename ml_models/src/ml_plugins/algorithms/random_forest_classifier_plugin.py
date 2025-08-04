import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
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

class RandomForestClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Random Forest Classifier Plugin - Robust Ensemble Method
    
    Random Forest builds multiple decision trees and combines their predictions
    through voting, reducing overfitting and improving generalization.
    It's one of the most popular and effective machine learning algorithms.
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
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=42,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None):
        """
        Initialize Random Forest Classifier with comprehensive parameter support
        
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
        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees
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
        self._name = "Random Forest"
        self._description = "Robust ensemble method that combines multiple decision trees to reduce overfitting and improve accuracy."
        self._category = "Tree-Based Models"
        self._algorithm_type = "Ensemble Classifier"
        self._paper_reference = "Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32."
        
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
            "ensemble_type": "Bagging (Bootstrap Aggregating)",
            "strengths": [
                "Reduces overfitting compared to single decision trees",
                "Handles missing values well (with preprocessing)",
                "Provides feature importance rankings",
                "Works well with default parameters",
                "Robust to outliers and noise",
                "Handles mixed data types naturally",
                "Parallel training capability",
                "Out-of-bag error estimation",
                "Generally excellent performance",
                "Requires minimal data preprocessing"
            ],
            "weaknesses": [
                "Can overfit with very noisy data",
                "Less interpretable than single decision trees",
                "Memory intensive with many trees",
                "Biased toward categorical variables with many categories",
                "Can be slow on very large datasets",
                "May not perform well on very high-dimensional sparse data"
            ],
            "use_cases": [
                "General-purpose classification problems",
                "Baseline model for most datasets",
                "Feature importance analysis",
                "Structured/tabular data problems",
                "When interpretability is moderately important",
                "Bioinformatics and genomics",
                "Finance and risk assessment",
                "Marketing and customer analytics",
                "Medical diagnosis",
                "Any domain with mixed feature types"
            ],
            "ensemble_details": {
                "base_learner": "Decision Trees",
                "combination_method": "Majority Voting",
                "sampling": "Bootstrap sampling with replacement",
                "feature_sampling": "Random subset of features per split",
                "variance_reduction": "Averaging reduces overfitting"
            },
            "complexity": {
                "training": "O(n × log(n) × m × k)",
                "prediction": "O(log(n) × k)",
                "space": "O(n × k)"
            },
            "parameters_guide": {
                "n_estimators": "More trees = better performance but slower (50-500)",
                "max_depth": "Controls individual tree complexity (3-20 or None)",
                "max_features": "sqrt: good default, log2: for high dimensions",
                "min_samples_split": "Prevents overfitting (2-20)",
                "bootstrap": "True: enables bagging, False: uses all data"
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Random Forest Classifier model
        
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
        
        # Create and configure the Random Forest model
        self.model_ = RandomForestClassifier(
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
            }
        }
        
        return diversity_info
    
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
        
        bars = ax.barh(range(len(top_features)), top_importance, color='skyblue', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {len(top_features)} Feature Importances - Random Forest')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🌲 Random Forest Configuration")
        
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
                help="More trees = better performance but slower training",
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
                help="sqrt: Good default, log2: For high dimensions, None: Use all features",
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
            
            # Bootstrap
            bootstrap = st.checkbox(
                "Bootstrap Sampling",
                value=self.bootstrap,
                help="Use bootstrap samples when building trees (enables bagging)",
                key=f"{key_prefix}_bootstrap"
            )
            
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
                "Limit Bootstrap Sample Size",
                value=self.max_samples is not None,
                help="Limit number of samples used per tree",
                key=f"{key_prefix}_max_samples_enabled"
            )
            
            if max_samples_enabled and bootstrap:
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
            
            # CCP Alpha (pruning)
            ccp_alpha = st.number_input(
                "Pruning Alpha (CCP):",
                value=float(self.ccp_alpha),
                min_value=0.0,
                max_value=0.1,
                step=0.001,
                format="%.4f",
                help="Complexity parameter for pruning individual trees",
                key=f"{key_prefix}_ccp_alpha"
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
            **Random Forest** excels at:
            • Reducing overfitting vs. single trees
            • Providing robust predictions
            • Handling mixed data types
            • Feature importance analysis
            • Working well with default parameters
            
            **Ensemble Magic:**
            • Combines many trees for better predictions
            • Each tree sees different data (bootstrap)
            • Each split considers random feature subset
            • Final prediction via majority voting
            """)
            
            # Performance tips
            if st.button("🚀 Performance Tips", key=f"{key_prefix}_performance_tips"):
                st.markdown("""
                **For Better Performance:**
                • **Start with defaults** - RF works well out-of-the-box
                • **Increase n_estimators** if you have time (100-500)
                • **Limit max_depth** if overfitting (5-15)
                • **Use sqrt(features)** for max_features (default)
                • **Enable OOB score** for free validation estimate
                • **Use n_jobs=-1** for parallel processing
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
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return RandomForestClassifierPlugin(
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
        
        Random Forest requires minimal preprocessing:
        1. Can handle mixed data types
        2. No scaling required
        3. Robust to outliers
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
                return False, f"Too many classes ({n_classes}). Random Forest works better with fewer classes."
            
            # Check for missing values (warning, not blocker for RF)
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
            
            # Random Forest specific advantages
            advantages = []
            considerations = []
            
            # Mixed data types (Random Forest's strength)
            if len(categorical_features) > 0 and len(numeric_features) > 0:
                advantages.append(f"Excellent for mixed data types ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
            
            # Dataset size advantages
            if n_samples >= 1000:
                advantages.append(f"Good dataset size ({n_samples} samples) for ensemble learning")
            elif n_samples < 100:
                considerations.append(f"Small dataset ({n_samples} samples) - may not fully benefit from ensemble")
            
            # Feature count
            if n_features >= 10:
                advantages.append(f"Good feature count ({n_features}) for Random Forest feature sampling")
            elif n_features < 5:
                considerations.append(f"Few features ({n_features}) - may not benefit from feature randomness")
            
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
            
            # Outlier robustness
            outlier_potential = False
            for col in numeric_features[:3]:  # Check first 3 numeric features
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    if outliers > 0:
                        outlier_potential = True
                        break
                except:
                    pass
            
            if outlier_potential:
                advantages.append("Robust to outliers (tree-based method)")
            
            # Class balance
            if target_column in df.columns:
                class_counts = df[target_column].value_counts()
                min_class_size = class_counts.min()
                max_class_size = class_counts.max()
                
                if max_class_size / min_class_size > 10:
                    considerations.append("Imbalanced classes detected - consider class_weight='balanced'")
                elif max_class_size / min_class_size < 3:
                    advantages.append("Well-balanced classes for ensemble learning")
            
            # Missing values handling
            if has_missing:
                if len(categorical_features) > 0:
                    return False, f"Missing values detected ({missing_values} total). Please handle missing values first - Random Forest requires complete data."
                else:
                    considerations.append(f"Missing values detected ({missing_values}) - will need preprocessing")
            
            # Categorical encoding needed
            if len(categorical_features) > 0:
                return False, f"Categorical features detected: {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}. Please encode categorical variables first."
            
            # Performance considerations
            if n_samples * n_features > 1000000:  # Large dataset
                considerations.append("Large dataset - consider reducing n_estimators or using n_jobs=-1 for parallel processing")
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if advantages:
                message_parts.append("🎯 Random Forest advantages: " + "; ".join(advantages))
            
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
            'ccp_alpha': self.ccp_alpha,
            'max_samples': self.max_samples
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        # Basic info
        info = {
            "status": "Fitted",
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
                    "gini_coefficient": float(self._calculate_gini_coefficient(feature_importance))
                }
            }
            info.update(top_features)
        
        return info
    
    def _calculate_gini_coefficient(self, importance):
        """Calculate Gini coefficient for feature importance concentration"""
        if len(importance) == 0:
            return 0.0
        
        # Sort the importance values
        sorted_importance = np.sort(importance)
        n = len(sorted_importance)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_importance)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_importance)) / (n * cumsum[-1]) - (n + 1) / n
        
        return max(0.0, gini)  # Ensure non-negative

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Random Forest Classifier model.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Not directly used for these specific metrics but kept for API consistency.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics but kept for API consistency.
        y_proba : np.ndarray, optional
            Predicted probabilities. Not directly used for these specific metrics but kept for API consistency.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve Random Forest specific metrics."}

        metrics = {}
        prefix = "rf_" # Prefix for Random Forest specific metrics

        # OOB Score
        if self.oob_score and hasattr(self.model_, 'oob_score_'):
            metrics[f"{prefix}oob_score"] = self.model_.oob_score_
        else:
            metrics[f"{prefix}oob_score"] = None # Explicitly state if not available/calculated

        # Feature Importances
        if hasattr(self.model_, 'feature_importances_'):
            importances = self.model_.feature_importances_
            metrics[f"{prefix}mean_feature_importance"] = float(np.mean(importances))
            metrics[f"{prefix}std_feature_importance"] = float(np.std(importances))
            metrics[f"{prefix}feature_importance_gini_coeff"] = float(self._calculate_gini_coefficient(importances))
            
            # Number of features with non-zero importance
            metrics[f"{prefix}num_important_features"] = int(np.sum(importances > 1e-6)) # Using a small threshold

        # Tree Diversity
        diversity_info = self.analyze_tree_diversity()
        if "n_trees" in diversity_info: # Check if analysis was successful
            metrics[f"{prefix}num_trees_built"] = diversity_info["n_trees"]
            
            if "depth_stats" in diversity_info:
                metrics[f"{prefix}mean_tree_depth"] = float(diversity_info["depth_stats"]["mean"])
                metrics[f"{prefix}std_tree_depth"] = float(diversity_info["depth_stats"]["std"])
                metrics[f"{prefix}max_tree_depth"] = int(diversity_info["depth_stats"]["max"])
                metrics[f"{prefix}min_tree_depth"] = int(diversity_info["depth_stats"]["min"])

            if "nodes_stats" in diversity_info:
                metrics[f"{prefix}mean_tree_nodes"] = float(diversity_info["nodes_stats"]["mean"])
                metrics[f"{prefix}std_tree_nodes"] = float(diversity_info["nodes_stats"]["std"])

            if "leaves_stats" in diversity_info:
                metrics[f"{prefix}mean_tree_leaves"] = float(diversity_info["leaves_stats"]["mean"])
                metrics[f"{prefix}std_tree_leaves"] = float(diversity_info["leaves_stats"]["std"])
        
        # Number of estimators configured vs. actual (if different, e.g. warm_start related)
        if hasattr(self.model_, 'estimators_'):
             metrics[f"{prefix}actual_n_estimators"] = len(self.model_.estimators_)


        if not metrics: # Should not happen if fitted, but as a fallback
            metrics['info'] = "No specific Random Forest metrics were available or calculated."
            
        return metrics

def get_plugin():
    """Factory function to get plugin instance"""
    return RandomForestClassifierPlugin()
