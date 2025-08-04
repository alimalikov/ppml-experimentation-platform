import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
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

class DecisionTreeClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Decision Tree Classifier Plugin - Highly Interpretable
    
    Decision trees are among the most interpretable machine learning algorithms,
    providing clear if-then rules that can be easily understood by humans.
    They handle both numerical and categorical features naturally and require
    minimal data preprocessing.
    """
    
    def __init__(self, 
                 criterion='gini',
                 splitter='best',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=42,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 ccp_alpha=0.0):
        """
        Initialize Decision Tree Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        criterion : {'gini', 'entropy', 'log_loss'}, default='gini'
            The function to measure the quality of a split
        splitter : {'best', 'random'}, default='best'
            The strategy used to choose the split at each node
        max_depth : int, default=None
            The maximum depth of the tree
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node
        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node
        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights
        max_features : int, float, str or None, default=None
            The number of features to consider when looking for the best split
        random_state : int, default=42
            Controls the randomness of the estimator
        max_leaf_nodes : int, default=None
            Grow a tree with max_leaf_nodes in best-first fashion
        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
        class_weight : dict, list of dict or 'balanced', default=None
            Weights associated with classes
        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning
        """
        super().__init__()
        
        # Algorithm parameters
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        
        # Plugin metadata
        self._name = "Decision Tree"
        self._description = "Highly interpretable tree-based classifier that creates human-readable if-then rules for predictions."
        self._category = "Tree-Based Models"
        self._algorithm_type = "Decision Tree"
        self._paper_reference = "Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 10
        self._handles_missing_values = False  # sklearn's DecisionTree doesn't handle missing values directly
        self._requires_scaling = False  # Decision trees are invariant to monotonic transformations
        self._supports_sparse = True
        self._is_linear = False
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = True  # Can handle categorical features well
        self._highly_interpretable = True
        self._supports_tree_visualization = True
        
        # Internal attributes
        self.model_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.tree_rules_ = None
        
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
            "interpretability": "Very High",
            "strengths": [
                "Highly interpretable - generates clear if-then rules",
                "No assumptions about data distribution",
                "Handles both numerical and categorical features",
                "No need for feature scaling",
                "Automatic feature selection",
                "Can capture non-linear relationships",
                "Fast training and prediction",
                "Robust to outliers in features (not target)",
                "Easy to visualize and explain"
            ],
            "weaknesses": [
                "Prone to overfitting (especially deep trees)",
                "Can be unstable (small data changes → different tree)",
                "Biased toward features with more levels",
                "Cannot extrapolate beyond training data",
                "May create overly complex trees",
                "Limited by greedy splitting algorithm",
                "Sensitive to class imbalance"
            ],
            "use_cases": [
                "Exploratory data analysis",
                "Rule extraction and explanation",
                "Medical diagnosis systems",
                "Credit scoring with transparency requirements",
                "Feature importance analysis",
                "Baseline model for complex datasets",
                "Educational purposes",
                "Regulatory environments requiring explainability"
            ],
            "interpretability_features": {
                "tree_structure": "Visual tree representation",
                "decision_rules": "If-then rules for each path",
                "feature_importance": "Gini/entropy-based importance",
                "leaf_statistics": "Class distributions at leaves",
                "pruning_analysis": "Cost-complexity pruning paths"
            },
            "complexity": {
                "training": "O(n × m × log(n))",
                "prediction": "O(log(n))",
                "space": "O(n)"
            },
            "parameters_guide": {
                "max_depth": "Controls overfitting - start with 3-7",
                "min_samples_split": "Prevents splitting small nodes - try 5-20",
                "min_samples_leaf": "Ensures meaningful leaves - try 2-10",
                "criterion": "gini: fast, entropy: more balanced trees"
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Decision Tree Classifier model
        
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
        
        # Create and configure the Decision Tree model
        self.model_ = DecisionTreeClassifier(
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
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha
        )
        
        # Train the model
        if sample_weight is not None:
            self.model_.fit(X, y_encoded, sample_weight=sample_weight)
        else:
            self.model_.fit(X, y_encoded)
        
        # Generate tree rules for interpretability
        self._generate_tree_rules()
        
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
    
    def decision_path(self, X):
        """
        Return the decision path in the tree
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        indicator : sparse matrix, shape (n_samples, n_nodes)
            Binary matrix indicating the decision path
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True, dtype=None)
        
        return self.model_.decision_path(X)
    
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
    
    def _generate_tree_rules(self):
        """Generate human-readable tree rules"""
        if not self.is_fitted_:
            return None
        
        try:
            self.tree_rules_ = export_text(
                self.model_, 
                feature_names=self.feature_names_,
                class_names=[str(cls) for cls in self.classes_]
            )
        except Exception:
            self.tree_rules_ = "Tree rules generation failed"
    
    def get_tree_rules(self) -> Optional[str]:
        """
        Get human-readable tree rules
        
        Returns:
        --------
        rules : str
            Text representation of the decision tree rules
        """
        return self.tree_rules_
    
    def get_tree_structure_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the tree structure
        
        Returns:
        --------
        info : dict
            Information about the tree structure
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        tree = self.model_.tree_
        
        info = {
            "n_nodes": tree.node_count,
            "n_leaves": tree.n_leaves,
            "max_depth": tree.max_depth,
            "n_features": tree.n_features,
            "n_classes": tree.n_classes[0],
            "total_samples": tree.n_node_samples[0],
            "tree_impurity": tree.impurity[0],
            "criterion_used": self.criterion
        }
        
        # Calculate tree complexity metrics
        info["average_depth"] = np.mean([tree.compute_node_depths()[i] for i in range(tree.node_count) if tree.children_left[i] == tree.children_right[i]])
        info["leaves_ratio"] = tree.n_leaves / tree.node_count if tree.node_count > 0 else 0
        
        return info
    
    def visualize_tree(self, max_depth_display=3, figsize=(15, 10)):
        """
        Create a visualization of the decision tree
        
        Parameters:
        -----------
        max_depth_display : int, default=3
            Maximum depth to display in visualization
        figsize : tuple, default=(15, 10)
            Figure size for the plot
            
        Returns:
        --------
        fig : matplotlib figure
            Tree visualization plot
        """
        if not self.is_fitted_:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_tree(
            self.model_,
            feature_names=self.feature_names_,
            class_names=[str(cls) for cls in self.classes_],
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=max_depth_display,
            ax=ax
        )
        
        ax.set_title(f"Decision Tree Visualization (Max Depth: {max_depth_display})", 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def get_leaf_predictions_explanation(self, X):
        """
        Get detailed explanation of predictions including decision paths
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to explain
            
        Returns:
        --------
        explanations : list
            List of explanation dictionaries for each sample
        """
        if not self.is_fitted_:
            return []
        
        X = check_array(X, accept_sparse=True, dtype=None)
        decision_paths = self.decision_path(X)
        leaf_ids = self.model_.apply(X)
        
        explanations = []
        
        for i in range(X.shape[0]):
            sample_path = decision_paths[i].toarray()[0]
            path_nodes = np.where(sample_path)[0]
            
            explanation = {
                "sample_index": i,
                "leaf_id": leaf_ids[i],
                "decision_path": [],
                "prediction": self.predict([X[i]])[0],
                "prediction_proba": self.predict_proba([X[i]])[0].tolist()
            }
            
            # Build decision path explanation
            tree = self.model_.tree_
            for node_id in path_nodes[:-1]:  # Exclude leaf node
                feature_id = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = self.feature_names_[feature_id]
                feature_value = X[i, feature_id]
                
                if X[i, feature_id] <= threshold:
                    decision = f"{feature_name} ({feature_value:.3f}) <= {threshold:.3f}"
                    direction = "left"
                else:
                    decision = f"{feature_name} ({feature_value:.3f}) > {threshold:.3f}"
                    direction = "right"
                
                explanation["decision_path"].append({
                    "node_id": int(node_id),
                    "feature": feature_name,
                    "feature_value": float(feature_value),
                    "threshold": float(threshold),
                    "decision": decision,
                    "direction": direction
                })
            
            explanations.append(explanation)
        
        return explanations
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🌳 Decision Tree Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3 = st.sidebar.tabs(["Core", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Criterion
            criterion = st.selectbox(
                "Split Criterion:",
                options=['gini', 'entropy', 'log_loss'],
                index=['gini', 'entropy', 'log_loss'].index(self.criterion),
                help="gini: Gini impurity, entropy: Information gain, log_loss: Logistic loss",
                key=f"{key_prefix}_criterion"
            )
            
            # Max depth
            max_depth_enabled = st.checkbox(
                "Limit Tree Depth",
                value=self.max_depth is not None,
                help="Prevent overfitting by limiting tree depth",
                key=f"{key_prefix}_max_depth_enabled"
            )
            
            if max_depth_enabled:
                max_depth = st.slider(
                    "Max Depth:",
                    min_value=1,
                    max_value=20,
                    value=int(self.max_depth) if self.max_depth else 5,
                    help="Maximum depth of the tree",
                    key=f"{key_prefix}_max_depth"
                )
            else:
                max_depth = None
            
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
            
            # Class weight
            class_weight_option = st.selectbox(
                "Class Weight:",
                options=['None', 'balanced'],
                index=0 if self.class_weight is None else 1,
                help="Balanced: Adjust weights inversely proportional to class frequencies",
                key=f"{key_prefix}_class_weight"
            )
            class_weight = None if class_weight_option == 'None' else 'balanced'
        
        with tab2:
            st.markdown("**Advanced Parameters**")
            
            # Splitter
            splitter = st.selectbox(
                "Splitter Strategy:",
                options=['best', 'random'],
                index=['best', 'random'].index(self.splitter),
                help="best: Best split, random: Random split (more regularization)",
                key=f"{key_prefix}_splitter"
            )
            
            # Max features
            max_features_option = st.selectbox(
                "Max Features:",
                options=['None', 'sqrt', 'log2', 'custom'],
                index=0,
                help="Maximum features to consider for best split",
                key=f"{key_prefix}_max_features_option"
            )
            
            if max_features_option == 'custom':
                max_features_value = st.slider(
                    "Number of Features:",
                    min_value=1,
                    max_value=20,  # Will be adjusted based on actual data
                    value=5,
                    key=f"{key_prefix}_max_features_value"
                )
                max_features = max_features_value
            elif max_features_option == 'None':
                max_features = None
            else:
                max_features = max_features_option
            
            # Max leaf nodes
            max_leaf_nodes_enabled = st.checkbox(
                "Limit Leaf Nodes",
                value=self.max_leaf_nodes is not None,
                help="Limit total number of leaf nodes",
                key=f"{key_prefix}_max_leaf_nodes_enabled"
            )
            
            if max_leaf_nodes_enabled:
                max_leaf_nodes = st.slider(
                    "Max Leaf Nodes:",
                    min_value=2,
                    max_value=100,
                    value=int(self.max_leaf_nodes) if self.max_leaf_nodes else 20,
                    key=f"{key_prefix}_max_leaf_nodes"
                )
            else:
                max_leaf_nodes = None
            
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
                help="Complexity parameter for minimal cost-complexity pruning",
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
            **Decision Trees** excel at:
            • Providing clear, interpretable rules
            • Handling mixed data types naturally
            • No need for feature scaling
            • Automatic feature selection
            • Non-linear pattern capture
            
            **Overfitting Prevention:**
            • Limit max_depth (start with 3-7)
            • Increase min_samples_split/leaf
            • Use pruning (ccp_alpha > 0)
            """)
            
            # Interpretability showcase
            if st.button("🔍 Interpretability Features", key=f"{key_prefix}_interpretability"):
                st.markdown("""
                **What makes Decision Trees interpretable:**
                • **Visual tree structure** - See the entire decision process
                • **If-then rules** - Each path is a simple rule
                • **Feature importance** - Know which features matter most
                • **Decision paths** - Trace any prediction step-by-step
                • **No black box** - Every decision is transparent
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": 0.0,
            "max_features": max_features,
            "random_state": random_state,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "class_weight": class_weight,
            "ccp_alpha": ccp_alpha
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return DecisionTreeClassifierPlugin(
            criterion=hyperparameters.get("criterion", self.criterion),
            splitter=hyperparameters.get("splitter", self.splitter),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            min_samples_split=hyperparameters.get("min_samples_split", self.min_samples_split),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", self.min_samples_leaf),
            min_weight_fraction_leaf=hyperparameters.get("min_weight_fraction_leaf", self.min_weight_fraction_leaf),
            max_features=hyperparameters.get("max_features", self.max_features),
            random_state=hyperparameters.get("random_state", self.random_state),
            max_leaf_nodes=hyperparameters.get("max_leaf_nodes", self.max_leaf_nodes),
            min_impurity_decrease=hyperparameters.get("min_impurity_decrease", self.min_impurity_decrease),
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            ccp_alpha=hyperparameters.get("ccp_alpha", self.ccp_alpha)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        Decision trees require minimal preprocessing:
        1. Can handle mixed data types
        2. No scaling required
        3. Can handle missing values with some techniques
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
                return False, f"Too many classes ({n_classes}). Decision trees work better with fewer classes."
            
            # Check for missing values (warning, not blocker)
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
            
            # Decision tree specific advantages
            advantages = []
            warnings_list = []
            
            # Mixed data types (Decision tree's strength)
            if len(categorical_features) > 0 and len(numeric_features) > 0:
                advantages.append(f"Handles mixed data types well ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
            elif len(categorical_features) > 0:
                advantages.append(f"Good for categorical data ({len(categorical_features)} categorical features)")
            
            # Interpretability advantage
            if n_features <= 20:
                advantages.append("Excellent interpretability with moderate feature count")
            elif n_features <= 50:
                advantages.append("Good interpretability, may need depth limiting")
            else:
                warnings_list.append(f"Many features ({n_features}) - tree may be complex, consider feature selection")
            
            # No scaling needed
            if len(numeric_features) > 0:
                # Check if features have different scales
                ranges = []
                for col in numeric_features[:5]:  # Check first 5 numeric features
                    try:
                        col_range = df[col].max() - df[col].min()
                        if col_range > 0:
                            ranges.append(col_range)
                    except:
                        pass
                
                if len(ranges) > 1 and max(ranges) / min(ranges) > 100:
                    advantages.append("No scaling required (robust to different feature scales)")
            
            # Dataset size considerations
            if n_samples < 100:
                warnings_list.append(f"Small dataset ({n_samples} samples) - risk of overfitting, use depth limiting")
            elif n_samples > 10000:
                advantages.append(f"Large dataset ({n_samples} samples) - tree can learn complex patterns")
            
            # Class balance
            if target_column in df.columns:
                class_counts = df[target_column].value_counts()
                min_class_size = class_counts.min()
                max_class_size = class_counts.max()
                
                if max_class_size / min_class_size > 10:
                    warnings_list.append("Imbalanced classes detected - consider class_weight='balanced'")
            
            # Missing values handling
            if has_missing:
                if len(categorical_features) > 0:
                    return False, f"Missing values detected ({missing_values} total). Please handle missing values first - decision trees in sklearn don't handle them automatically."
                else:
                    warnings_list.append(f"Missing values detected ({missing_values}) - will need preprocessing")
            
            # Categorical encoding needed
            if len(categorical_features) > 0:
                return False, f"Categorical features detected: {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}. Please encode categorical variables first (Label Encoding or One-Hot Encoding)."
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if advantages:
                message_parts.append("🎯 Decision Tree advantages: " + "; ".join(advantages))
            
            if warnings_list:
                message_parts.append("⚠️ Considerations: " + "; ".join(warnings_list))
            
            return True, " | ".join(message_parts)
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'criterion': self.criterion,
            'splitter': self.splitter,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'class_weight': self.class_weight,
            'ccp_alpha': self.ccp_alpha
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
            "feature_names": self.feature_names_
        }
        
        # Tree structure info
        tree_info = self.get_tree_structure_info()
        info.update(tree_info)
        
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
                    for idx in top_features_idx if feature_importance[idx] > 0
                ]
            }
            info.update(top_features)
        
        return info

    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Decision Tree Classifier-specific metrics.

        Args:
            y_true: Ground truth target values. (Not directly used for tree structure metrics)
            y_pred: Predicted target values. (Not directly used for tree structure metrics)
            y_proba: Predicted probabilities. (Not directly used for tree structure metrics)

        Returns:
            A dictionary of Decision Tree Classifier-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted"
            return metrics

        # --- Metrics from tree structure ---
        tree_structure_info = self.get_tree_structure_info()
        if tree_structure_info.get("status") != "Not fitted":
            metrics['tree_n_nodes'] = tree_structure_info.get('n_nodes')
            metrics['tree_n_leaves'] = tree_structure_info.get('n_leaves')
            metrics['tree_max_depth_actual'] = tree_structure_info.get('max_depth')
            metrics['tree_root_impurity'] = tree_structure_info.get('tree_impurity')
            metrics['tree_average_leaf_depth'] = tree_structure_info.get('average_depth') # Renamed for clarity
            metrics['tree_leaves_to_nodes_ratio'] = tree_structure_info.get('leaves_ratio') # Renamed for clarity

        metrics['tree_criterion_used'] = self.model_.criterion # Or self.criterion, should be same after fit

        # --- Metrics from feature importance ---
        feature_importances = self.get_feature_importance()
        if feature_importances is not None:
            metrics['num_features_used_in_tree'] = int(np.sum(feature_importances > 0))
            if len(feature_importances) > 0:
                metrics['max_feature_importance'] = float(np.max(feature_importances))
                # Gini coefficient of feature importances (concentration)
                sorted_importances = np.sort(feature_importances)
                n_features = len(sorted_importances)
                if np.sum(sorted_importances) > 0 and n_features > 0:
                    index = np.arange(1, n_features + 1)
                    gini_importance_concentration = (2 * np.sum(index * sorted_importances)) / (n_features * np.sum(sorted_importances)) - (n_features + 1) / n_features
                    metrics['feature_importance_gini_concentration'] = float(gini_importance_concentration)
                else:
                    metrics['feature_importance_gini_concentration'] = 0.0 if n_features > 0 else np.nan


        # --- Pruning parameter used ---
        metrics['ccp_alpha_applied'] = self.ccp_alpha # The value used for training

        return metrics

def get_plugin():
    """Factory function to get plugin instance"""
    return DecisionTreeClassifierPlugin()