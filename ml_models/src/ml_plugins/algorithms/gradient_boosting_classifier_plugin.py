import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import validation_curve
from sklearn.metrics import log_loss # Add this import
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

class GradientBoostingClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Gradient Boosting Classifier Plugin - High Performance Sequential Ensemble
    
    Gradient Boosting builds models sequentially, where each new model corrects
    errors made by previous models. It's one of the most powerful and widely-used
    machine learning algorithms, often achieving state-of-the-art results.
    """
    
    def __init__(self, 
                 loss='log_loss',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,
                 min_impurity_decrease=0.0,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 ccp_alpha=0.0,
                 random_state=42):
        """
        Initialize Gradient Boosting Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        loss : {'log_loss', 'exponential'}, default='log_loss'
            Loss function to be optimized
        learning_rate : float, default=0.1
            Shrinkage rate that controls the contribution of each tree
        n_estimators : int, default=100
            Number of boosting stages to perform
        subsample : float, default=1.0
            Fraction of samples used for fitting individual base learners
        criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
            Function to measure the quality of a split
        min_samples_split : int or float, default=2
            Minimum number of samples required to split an internal node
        min_samples_leaf : int or float, default=1
            Minimum number of samples required to be at a leaf node
        min_weight_fraction_leaf : float, default=0.0
            Minimum weighted fraction of the sum total of weights
        max_depth : int, default=3
            Maximum depth of the individual regression estimators
        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
        max_features : {'sqrt', 'log2', None}, int or float, default=None
            Number of features to consider when looking for the best split
        alpha : float, default=0.9
            Alpha-quantile of the huber loss function and quantile loss
        verbose : int, default=0
            Enable verbose output
        max_leaf_nodes : int, default=None
            Grow trees with max_leaf_nodes in best-first fashion
        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit
        validation_fraction : float, default=0.1
            Proportion of training data to set aside as validation set
        n_iter_no_change : int, default=None
            Number of iterations with no improvement to wait before early stopping
        tol : float, default=1e-4
            Tolerance for the early stopping
        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__()
        
        # Algorithm parameters
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        
        # Plugin metadata
        self._name = "Gradient Boosting"
        self._description = "High-performance sequential ensemble that builds models iteratively to correct previous errors. Excellent for competitions."
        self._category = "Tree-Based Models"
        self._algorithm_type = "Sequential Ensemble Classifier"
        self._paper_reference = "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 50
        self._handles_missing_values = False
        self._requires_scaling = False
        self._supports_sparse = False
        self._is_linear = False
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = True
        self._ensemble_method = True
        self._sequential_learning = True
        self._supports_early_stopping = True
        self._supports_validation_monitoring = True
        self._high_performance = True
        self._competition_grade = True
        
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
            "ensemble_type": "Sequential Boosting",
            "learning_paradigm": {
                "approach": "Sequential Error Correction",
                "base_learner": "Decision Trees (typically shallow)",
                "combination": "Weighted sum with learned coefficients",
                "optimization": "Gradient descent in function space",
                "loss_minimization": "Iterative residual fitting"
            },
            "strengths": [
                "Excellent predictive performance",
                "Handles mixed data types naturally",
                "Provides feature importance rankings",
                "Robust to outliers (with appropriate loss functions)",
                "Good bias-variance tradeoff",
                "No need for data preprocessing",
                "Handles missing values implicitly",
                "Sequential learning captures complex patterns",
                "Built-in regularization through shrinkage",
                "Early stopping prevents overfitting",
                "Validation monitoring for optimal iterations"
            ],
            "weaknesses": [
                "Can easily overfit with too many iterations",
                "Sensitive to hyperparameters",
                "Sequential nature makes it slower to train",
                "No natural parallelization (unlike Random Forest)",
                "Requires careful tuning of learning rate",
                "Memory intensive for large datasets",
                "Can be sensitive to noisy data",
                "More complex to interpret than single trees"
            ],
            "use_cases": [
                "Kaggle competitions and machine learning contests",
                "Structured/tabular data problems",
                "When high predictive accuracy is crucial",
                "Feature importance analysis",
                "Ranking and scoring problems",
                "Medical diagnosis and prognosis",
                "Financial risk assessment",
                "Marketing response prediction",
                "Quality control and defect detection",
                "Any domain requiring top-tier performance"
            ],
            "algorithmic_details": {
                "boosting_process": [
                    "Initialize with simple prediction (class probabilities)",
                    "For each iteration:",
                    "  - Calculate residuals (gradients of loss)",
                    "  - Fit weak learner to residuals",
                    "  - Find optimal step size (line search)",
                    "  - Update ensemble prediction",
                    "Repeat until convergence or max iterations"
                ],
                "key_components": {
                    "weak_learners": "Shallow decision trees (depth 3-8)",
                    "loss_function": "Differentiable loss (log-loss, exponential)",
                    "learning_rate": "Shrinkage factor for regularization",
                    "gradient_computation": "First-order optimization"
                },
                "regularization_techniques": [
                    "Learning rate shrinkage",
                    "Subsampling (stochastic gradient boosting)",
                    "Tree depth limitation",
                    "Early stopping on validation set",
                    "Min samples per leaf constraints"
                ]
            },
            "complexity": {
                "training": "O(n × log(n) × m × T × d)",
                "prediction": "O(T × d)",
                "space": "O(T × d)"
            },
            "hyperparameter_guide": {
                "n_estimators": "Start with 100, increase if underfitting (100-1000)",
                "learning_rate": "Lower is better but slower (0.01-0.3)",
                "max_depth": "Keep shallow for better generalization (3-8)",
                "subsample": "0.8-1.0, lower values add regularization",
                "min_samples_leaf": "Higher values prevent overfitting (1-20)"
            },
            "tuning_strategy": {
                "step1": "Fix n_estimators=100, tune learning_rate and max_depth",
                "step2": "Increase n_estimators, lower learning_rate",
                "step3": "Tune min_samples_split and min_samples_leaf",
                "step4": "Add subsample if overfitting",
                "step5": "Enable early stopping for optimal iterations"
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Gradient Boosting Classifier model
        
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
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
        
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
        
        # Create and configure the Gradient Boosting model
        self.model_ = GradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            alpha=self.alpha,
            verbose=self.verbose,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            ccp_alpha=self.ccp_alpha,
            random_state=self.random_state
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
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
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
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        return self.model_.predict_proba(X)
    
    def predict_log_proba(self, X):
        """
        Predict log probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        log_probabilities : array, shape (n_samples, n_classes)
            Log probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        return self.model_.predict_log_proba(X)
    
    def decision_function(self, X):
        """
        Predict confidence scores for samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        scores : array, shape (n_samples,) or (n_samples, n_classes)
            Confidence scores
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        return self.model_.decision_function(X)
    
    def staged_predict(self, X):
        """
        Predict class labels at each stage of boosting
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        predictions : generator of arrays
            Predictions at each boosting stage
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        for stage_pred in self.model_.staged_predict(X):
            yield self.label_encoder_.inverse_transform(stage_pred)
    
    def staged_predict_proba(self, X):
        """
        Predict class probabilities at each stage of boosting
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        probabilities : generator of arrays
            Probabilities at each boosting stage
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        return self.model_.staged_predict_proba(X)
    
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
    
    def get_training_loss(self) -> Optional[np.ndarray]:
        """
        Get training loss at each boosting iteration
        
        Returns:
        --------
        training_loss : array, shape (n_estimators,)
            Training loss at each iteration
        """
        if not self.is_fitted_:
            return None
            
        return self.model_.train_score_
    
    def get_validation_loss(self) -> Optional[np.ndarray]:
        """
        Get validation loss at each boosting iteration (if validation_fraction > 0)
        
        Returns:
        --------
        validation_loss : array, shape (n_estimators,)
            Validation loss at each iteration
        """
        if not self.is_fitted_:
            return None
            
        return getattr(self.model_, 'validation_score_', None)
    
    def get_boosting_analysis(self) -> Dict[str, Any]:
        """
        Analyze the boosting process and convergence
        
        Returns:
        --------
        boosting_info : dict
            Information about the boosting process
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "n_estimators_fitted": len(self.model_.estimators_),
            "n_estimators_requested": self.n_estimators,
            "learning_rate": self.learning_rate,
            "converged": hasattr(self.model_, 'n_estimators_') and self.model_.n_estimators_ < self.n_estimators
        }
        
        # Training loss analysis
        train_loss = self.get_training_loss()
        if train_loss is not None:
            analysis["training_loss"] = {
                "final_loss": float(train_loss[-1]),
                "initial_loss": float(train_loss[0]),
                "improvement": float(train_loss[0] - train_loss[-1]),
                "improvement_pct": float((train_loss[0] - train_loss[-1]) / train_loss[0] * 100),
                "loss_trajectory": train_loss.tolist()
            }
        
        # Validation loss analysis
        val_loss = self.get_validation_loss()
        if val_loss is not None:
            analysis["validation_loss"] = {
                "final_loss": float(val_loss[-1]),
                "initial_loss": float(val_loss[0]),
                "improvement": float(val_loss[0] - val_loss[-1]),
                "improvement_pct": float((val_loss[0] - val_loss[-1]) / val_loss[0] * 100),
                "best_iteration": int(np.argmin(val_loss)),
                "overfitting_detected": len(val_loss) > 10 and val_loss[-1] > np.min(val_loss),
                "loss_trajectory": val_loss.tolist()
            }
        
        # Early stopping analysis
        if hasattr(self.model_, 'n_estimators_'):
            analysis["early_stopping"] = {
                "triggered": True,
                "stopped_at": self.model_.n_estimators_,
                "requested": self.n_estimators,
                "iterations_saved": self.n_estimators - self.model_.n_estimators_
            }
        else:
            analysis["early_stopping"] = {
                "triggered": False,
                "completed_all": True
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
                      color='darkblue', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {len(top_features)} Feature Importances - Gradient Boosting')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, figsize=(12, 8)):
        """
        Plot training and validation loss curves
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Learning curves plot
        """
        if not self.is_fitted_:
            return None
        
        train_loss = self.get_training_loss()
        val_loss = self.get_validation_loss()
        
        if train_loss is None:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        iterations = range(1, len(train_loss) + 1)
        
        # Training loss curve
        ax1.plot(iterations, train_loss, 'b-', label='Training Loss', linewidth=2)
        if val_loss is not None:
            ax1.plot(iterations, val_loss, 'r-', label='Validation Loss', linewidth=2)
            ax1.legend()
        ax1.set_xlabel('Boosting Iterations')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Curves - Loss vs Iterations')
        ax1.grid(True, alpha=0.3)
        
        # Loss improvement rate
        if len(train_loss) > 1:
            train_improvement = np.diff(train_loss)
            ax2.plot(iterations[1:], train_improvement, 'b-', label='Training Loss Change', alpha=0.7)
            if val_loss is not None and len(val_loss) > 1:
                val_improvement = np.diff(val_loss)
                ax2.plot(iterations[1:], val_improvement, 'r-', label='Validation Loss Change', alpha=0.7)
                ax2.legend()
            ax2.set_xlabel('Boosting Iterations')
            ax2.set_ylabel('Loss Change')
            ax2.set_title('Loss Improvement Rate')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Overfitting detection
        if val_loss is not None:
            overfitting_point = np.argmin(val_loss)
            ax3.plot(iterations, train_loss, 'b-', label='Training Loss', linewidth=2)
            ax3.plot(iterations, val_loss, 'r-', label='Validation Loss', linewidth=2)
            ax3.axvline(x=overfitting_point + 1, color='green', linestyle='--', 
                       label=f'Best Iteration ({overfitting_point + 1})', alpha=0.7)
            ax3.set_xlabel('Boosting Iterations')
            ax3.set_ylabel('Loss')
            ax3.set_title('Overfitting Detection')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Convergence analysis
        if len(train_loss) > 10:
            # Moving average for smoothness
            window = min(10, len(train_loss) // 4)
            train_smooth = np.convolve(train_loss, np.ones(window)/window, mode='valid')
            smooth_iterations = range(window, len(train_loss) + 1)
            
            ax4.plot(iterations, train_loss, 'b-', alpha=0.5, label='Training Loss')
            ax4.plot(smooth_iterations, train_smooth, 'darkblue', linewidth=2, label='Smoothed Training')
            
            if val_loss is not None:
                val_smooth = np.convolve(val_loss, np.ones(window)/window, mode='valid')
                ax4.plot(iterations, val_loss, 'r-', alpha=0.5, label='Validation Loss')
                ax4.plot(smooth_iterations, val_smooth, 'darkred', linewidth=2, label='Smoothed Validation')
            
            ax4.set_xlabel('Boosting Iterations')
            ax4.set_ylabel('Loss')
            ax4.set_title('Convergence Analysis (Smoothed)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_boosting_analysis(self, figsize=(12, 6)):
        """
        Create comprehensive boosting analysis plots
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Boosting analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        analysis = self.get_boosting_analysis()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Loss trajectory comparison
        if 'training_loss' in analysis and 'loss_trajectory' in analysis['training_loss']:
            train_loss = analysis['training_loss']['loss_trajectory']
            iterations = range(1, len(train_loss) + 1)
            
            ax1.plot(iterations, train_loss, 'b-', linewidth=2, label='Training Loss')
            
            if 'validation_loss' in analysis and 'loss_trajectory' in analysis['validation_loss']:
                val_loss = analysis['validation_loss']['loss_trajectory']
                ax1.plot(iterations, val_loss, 'r-', linewidth=2, label='Validation Loss')
                
                # Mark best iteration
                if 'best_iteration' in analysis['validation_loss']:
                    best_iter = analysis['validation_loss']['best_iteration']
                    ax1.axvline(x=best_iter + 1, color='green', linestyle='--', 
                               alpha=0.7, label=f'Best Iteration ({best_iter + 1})')
            
            ax1.set_xlabel('Boosting Iterations')
            ax1.set_ylabel('Loss')
            ax1.set_title('Boosting Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Performance metrics summary
        metrics_data = []
        labels = []
        
        if 'training_loss' in analysis:
            metrics_data.append(analysis['training_loss'].get('improvement_pct', 0))
            labels.append('Training\nImprovement (%)')
        
        if 'validation_loss' in analysis:
            metrics_data.append(analysis['validation_loss'].get('improvement_pct', 0))
            labels.append('Validation\nImprovement (%)')
        
        if analysis.get('n_estimators_fitted', 0) > 0:
            completion_pct = (analysis['n_estimators_fitted'] / analysis['n_estimators_requested']) * 100
            metrics_data.append(completion_pct)
            labels.append('Completion (%)')
        
        if metrics_data:
            colors = ['blue', 'red', 'green'][:len(metrics_data)]
            bars = ax2.bar(labels, metrics_data, color=colors, alpha=0.7)
            ax2.set_ylabel('Percentage')
            ax2.set_title('Boosting Summary Metrics')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_data):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🚀 Gradient Boosting Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4 = st.sidebar.tabs(["Core", "Trees", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Boosting Parameters**")
            
            # Learning rate
            learning_rate = st.number_input(
                "Learning Rate:",
                value=float(self.learning_rate),
                min_value=0.001,
                max_value=1.0,
                step=0.01,
                help="Shrinkage rate. Lower = better generalization but slower convergence",
                key=f"{key_prefix}_learning_rate"
            )
            
            # Number of estimators
            n_estimators = st.slider(
                "Number of Estimators:",
                min_value=10,
                max_value=1000,
                value=int(self.n_estimators),
                step=10,
                help="Number of boosting stages. More = better fit but risk of overfitting",
                key=f"{key_prefix}_n_estimators"
            )
            
            # Loss function
            loss_options = ['log_loss', 'exponential']
            loss = st.selectbox(
                "Loss Function:",
                options=loss_options,
                index=loss_options.index(self.loss),
                help="log_loss: Standard choice, exponential: AdaBoost-like",
                key=f"{key_prefix}_loss"
            )
            
            # Subsample
            subsample = st.slider(
                "Subsample Ratio:",
                min_value=0.1,
                max_value=1.0,
                value=float(self.subsample),
                step=0.05,
                help="Fraction of samples used per iteration. <1.0 adds stochasticity",
                key=f"{key_prefix}_subsample"
            )
            
            # Validation fraction
            validation_fraction = st.slider(
                "Validation Fraction:",
                min_value=0.0,
                max_value=0.3,
                value=float(self.validation_fraction),
                step=0.05,
                help="Fraction of training data for early stopping validation",
                key=f"{key_prefix}_validation_fraction"
            )
        
        with tab2:
            st.markdown("**Individual Tree Parameters**")
            
            # Max depth
            max_depth = st.slider(
                "Max Tree Depth:",
                min_value=1,
                max_value=15,
                value=int(self.max_depth),
                help="Depth of individual trees. Gradient Boosting works well with shallow trees",
                key=f"{key_prefix}_max_depth"
            )
            
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
            
            # Max features
            max_features_option = st.selectbox(
                "Max Features per Split:",
                options=['None', 'sqrt', 'log2', 'custom'],
                index=0 if self.max_features is None else 
                      1 if self.max_features == 'sqrt' else 
                      2 if self.max_features == 'log2' else 3,
                help="None: Use all features, sqrt: Good for noisy data",
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
            
            # Criterion
            criterion = st.selectbox(
                "Split Criterion:",
                options=['friedman_mse', 'squared_error'],
                index=['friedman_mse', 'squared_error'].index(self.criterion),
                help="friedman_mse: Usually better for boosting, squared_error: Standard MSE",
                key=f"{key_prefix}_criterion"
            )
        
        with tab3:
            st.markdown("**Advanced Parameters**")
            
            # Early stopping
            early_stopping_enabled = st.checkbox(
                "Enable Early Stopping",
                value=self.n_iter_no_change is not None,
                help="Stop when validation score stops improving",
                key=f"{key_prefix}_early_stopping_enabled"
            )
            
            if early_stopping_enabled:
                n_iter_no_change = st.slider(
                    "Patience (iterations):",
                    min_value=5,
                    max_value=50,
                    value=int(self.n_iter_no_change) if self.n_iter_no_change else 10,
                    help="Number of iterations with no improvement before stopping",
                    key=f"{key_prefix}_n_iter_no_change"
                )
                
                tol = st.number_input(
                    "Tolerance:",
                    value=float(self.tol),
                    min_value=1e-6,
                    max_value=1e-2,
                    step=1e-6,
                    format="%.2e",
                    help="Minimum improvement required",
                    key=f"{key_prefix}_tol"
                )
            else:
                n_iter_no_change = None
                tol = self.tol
            
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
            
            # Warm start
            warm_start = st.checkbox(
                "Warm Start",
                value=self.warm_start,
                help="Reuse previous model when fitting (allows incremental training)",
                key=f"{key_prefix}_warm_start"
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
        
        with tab4:
            st.markdown("**Algorithm Information**")
            st.info("""
            **Gradient Boosting** excels at:
            • Achieving excellent predictive performance
            • Sequential error correction
            • Handling complex non-linear patterns
            • Feature importance analysis
            • Competition-grade results
            
            **Sequential Learning:**
            • Each tree corrects previous errors
            • Gradient-based optimization
            • Built-in regularization via shrinkage
            • Early stopping prevents overfitting
            """)
            
            # Hyperparameter tuning guide
            if st.button("🎯 Tuning Strategy", key=f"{key_prefix}_tuning_guide"):
                st.markdown("""
                **Gradient Boosting Tuning Strategy:**
                
                **Step 1: Basic Setup**
                - Start: n_estimators=100, learning_rate=0.1, max_depth=3
                - Focus on learning_rate and max_depth first
                
                **Step 2: Learning Rate**
                - Lower learning_rate → better performance
                - Compensate with higher n_estimators
                - Rule: learning_rate × n_estimators ≈ constant
                
                **Step 3: Tree Complexity**
                - max_depth: 3-8 (shallow trees work best)
                - min_samples_leaf: 1-20 (higher prevents overfitting)
                
                **Step 4: Regularization**
                - subsample: 0.8-1.0 (adds stochasticity)
                - Enable early stopping for optimal iterations
                """)
            
            # Performance tips
            if st.button("⚡ Performance Tips", key=f"{key_prefix}_performance_tips"):
                st.markdown("""
                **For Better Performance:**
                • **Lower learning_rate** (0.01-0.1) with more estimators
                • **Shallow trees** (max_depth=3-6) prevent overfitting  
                • **Enable early stopping** to find optimal iterations
                • **Use validation monitoring** to track overfitting
                • **Subsample < 1.0** adds regularization
                • **Feature selection** can improve speed and performance
                """)
            
            # Competition tips
            if st.button("🏆 Competition Tips", key=f"{key_prefix}_competition_tips"):
                st.markdown("""
                **Kaggle/Competition Strategy:**
                
                **For Maximum Performance:**
                1. **Ensemble multiple GB models** with different random_state
                2. **Tune learning_rate + n_estimators** together
                3. **Use cross-validation** for hyperparameter selection
                4. **Feature engineering** is crucial for GB success
                5. **Combine with other algorithms** (XGBoost, LightGBM)
                
                **Typical Winning Settings:**
                - learning_rate: 0.01-0.05
                - n_estimators: 500-2000
                - max_depth: 4-8
                - subsample: 0.8-0.9
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "loss": loss,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "criterion": criterion,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": 0.0,
            "max_depth": max_depth,
            "min_impurity_decrease": min_impurity_decrease,
            "max_features": max_features,
            "alpha": 0.9,
            "verbose": 0,
            "max_leaf_nodes": None,
            "warm_start": warm_start,
            "validation_fraction": validation_fraction,
            "n_iter_no_change": n_iter_no_change,
            "tol": tol,
            "ccp_alpha": ccp_alpha,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return GradientBoostingClassifierPlugin(
            loss=hyperparameters.get("loss", self.loss),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            n_estimators=hyperparameters.get("n_estimators", self.n_estimators),
            subsample=hyperparameters.get("subsample", self.subsample),
            criterion=hyperparameters.get("criterion", self.criterion),
            min_samples_split=hyperparameters.get("min_samples_split", self.min_samples_split),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", self.min_samples_leaf),
            min_weight_fraction_leaf=hyperparameters.get("min_weight_fraction_leaf", self.min_weight_fraction_leaf),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            min_impurity_decrease=hyperparameters.get("min_impurity_decrease", self.min_impurity_decrease),
            max_features=hyperparameters.get("max_features", self.max_features),
            alpha=hyperparameters.get("alpha", self.alpha),
            verbose=hyperparameters.get("verbose", self.verbose),
            max_leaf_nodes=hyperparameters.get("max_leaf_nodes", self.max_leaf_nodes),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            validation_fraction=hyperparameters.get("validation_fraction", self.validation_fraction),
            n_iter_no_change=hyperparameters.get("n_iter_no_change", self.n_iter_no_change),
            tol=hyperparameters.get("tol", self.tol),
            ccp_alpha=hyperparameters.get("ccp_alpha", self.ccp_alpha),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        Gradient Boosting requires minimal preprocessing:
        1. Can handle mixed data types
        2. No scaling required
        3. Robust to outliers
        4. Handles missing values implicitly
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
                return False, f"Minimum {self._min_samples_required} samples required for reliable boosting, got {n_samples}"
            
            # Check target variable type
            target_values = df[target_column].unique()
            n_classes = len(target_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). Gradient Boosting works better with fewer classes."
            
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
            
            # Gradient Boosting specific advantages
            advantages = []
            considerations = []
            
            # Competition-grade performance
            advantages.append("Competition-grade algorithm with excellent predictive performance")
            
            # Sequential learning advantage
            if n_features >= 10:
                advantages.append(f"Good feature count ({n_features}) for sequential error correction")
            
            # Dataset size advantages
            if n_samples >= 1000:
                advantages.append(f"Good dataset size ({n_samples} samples) for stable boosting")
            elif n_samples < 200:
                considerations.append(f"Small dataset ({n_samples} samples) - consider fewer estimators and higher learning rate")
            
            # Class balance
            if target_column in df.columns:
                class_counts = df[target_column].value_counts()
                min_class_size = class_counts.min()
                max_class_size = class_counts.max()
                
                if max_class_size / min_class_size > 10:
                    considerations.append("Imbalanced classes detected - Gradient Boosting handles this well naturally")
                else:
                    advantages.append("Well-balanced classes ideal for boosting")
            
            # Feature type handling
            if len(categorical_features) > 0 and len(numeric_features) > 0:
                advantages.append(f"Excellent for mixed data types ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
            
            # High-dimensional data
            if n_features > n_samples / 10:
                considerations.append(f"High-dimensional case ({n_features} features) - consider max_features parameter and regularization")
            
            # Overfitting potential
            if n_features > 50 or n_samples < 500:
                considerations.append("Consider early stopping and validation monitoring to prevent overfitting")
            
            # Performance considerations
            if n_samples >= 10000:
                considerations.append("Large dataset - boosting may be slower than parallel methods (Random Forest)")
                advantages.append("Large dataset - consider subsample parameter for stochastic boosting")
            
            # Missing values handling
            if has_missing:
                if len(categorical_features) > 0:
                    return False, f"Missing values detected ({missing_values} total). Please handle missing values first - Gradient Boosting requires complete data."
                else:
                    considerations.append(f"Missing values detected ({missing_values}) - will need preprocessing")
            
            # Categorical encoding needed
            if len(categorical_features) > 0:
                return False, f"Categorical features detected: {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}. Please encode categorical variables first."
            
            # Performance optimization suggestions
            optimization_tips = []
            if n_samples > 5000:
                optimization_tips.append("enable early stopping for faster training")
            if n_features > 20:
                optimization_tips.append("consider max_features for regularization")
            if n_samples < 1000:
                optimization_tips.append("use higher learning_rate (0.1-0.3) for faster convergence")
            
            if optimization_tips:
                considerations.append(f"Optimization tips: {', '.join(optimization_tips)}")
            
            # Tuning recommendations
            tuning_recs = []
            if n_samples >= 1000:
                tuning_recs.append("tune learning_rate + n_estimators together")
            if n_classes > 2:
                tuning_recs.append("multiclass problem - may need more estimators")
            
            if tuning_recs:
                advantages.append(f"Tuning recommendations: {', '.join(tuning_recs)}")
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if advantages:
                message_parts.append("🚀 Gradient Boosting advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'criterion': self.criterion,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'random_state': self.random_state
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        # Basic info
        info = {
            "status": "Fitted",
            "algorithm": "Gradient Boosting (Sequential Ensemble)",
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": list(self.classes_),
            "feature_names": self.feature_names_,
            "n_estimators_configured": self.n_estimators # Renamed for clarity vs n_estimators_used
        }
        
        # Boosting analysis
        boosting_info = self.get_boosting_analysis()
        info["boosting_analysis"] = boosting_info
        if "n_estimators_fitted" in boosting_info:
            info["n_estimators_used"] = boosting_info["n_estimators_fitted"]
        
        # Feature importance
        feature_importances_array = self.get_feature_importance() # Corrected variable name
        if feature_importances_array is not None:
            # Get top 5 most important features
            top_features_idx = np.argsort(feature_importances_array)[-5:][::-1]
            top_features_summary = { # Renamed for clarity
                "top_features_summary": [ # Renamed for clarity
                    {
                        "feature": self.feature_names_[idx],
                        "importance": float(feature_importances_array[idx])
                    }
                    for idx in top_features_idx
                ],
                "importance_concentration": {
                    "top_5_sum": float(np.sum(feature_importances_array[top_features_idx])),
                    "max_importance": float(np.max(feature_importances_array)),
                    "min_importance": float(np.min(feature_importances_array))
                }
            }
            info.update(top_features_summary) # Use the renamed variable
        
        return info

    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Gradient Boosting Classifier-specific metrics.

        These metrics are derived from the model's internal state after fitting
        (e.g., boosting process, feature importances) and can also include
        metrics like log_loss if y_true and y_proba are provided for an
        external dataset.

        Args:
            y_true: Ground truth target values.
            y_pred: Predicted target values.
            y_proba: Predicted probabilities.

        Returns:
            A dictionary of algorithm-specific metrics.
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
            # Ensure we don't return a list/array if a scalar is expected
            if isinstance(current, (list, np.ndarray)) and len(current) > 10: # Heuristic for trajectories
                 return default # Or handle appropriately if list is expected
            return current if pd.notna(current) and not isinstance(current, (list, np.ndarray)) else default


        # --- Metrics from Boosting Analysis ---
        boosting_analysis = self.get_boosting_analysis()
        if boosting_analysis.get("status") != "Not fitted":
            metrics['n_estimators_used'] = safe_get(boosting_analysis, 'n_estimators_fitted')
            metrics['early_stopping_triggered'] = safe_get(boosting_analysis, 'early_stopping.triggered', False)
            if metrics['early_stopping_triggered']:
                 metrics['iterations_saved_by_early_stopping'] = safe_get(boosting_analysis, 'early_stopping.iterations_saved')

            metrics['final_training_loss'] = safe_get(boosting_analysis, 'training_loss.final_loss')
            metrics['training_loss_improvement_pct'] = safe_get(boosting_analysis, 'training_loss.improvement_pct')

            if 'validation_loss' in boosting_analysis:
                metrics['final_validation_loss'] = safe_get(boosting_analysis, 'validation_loss.final_loss')
                metrics['validation_loss_improvement_pct'] = safe_get(boosting_analysis, 'validation_loss.improvement_pct')
                metrics['best_iteration_validation'] = safe_get(boosting_analysis, 'validation_loss.best_iteration')
                metrics['overfitting_detected_by_validation'] = safe_get(boosting_analysis, 'validation_loss.overfitting_detected', False)

        # --- Metrics from Feature Importance ---
        importances = self.get_feature_importance()
        if importances is not None and len(importances) > 0:
            metrics['num_important_features'] = np.sum(importances > 0)
            metrics['max_feature_importance'] = np.max(importances)
            metrics['mean_feature_importance'] = np.mean(importances)
            
            # Gini coefficient for feature importance concentration
            if len(importances) > 1:
                sorted_importances = np.sort(importances)
                n = len(sorted_importances)
                cum_importances = np.cumsum(sorted_importances)
                # Area under Lorenz curve
                lorenz_area = cum_importances.sum() / (n * cum_importances[-1]) if cum_importances[-1] > 0 else 0.5
                metrics['feature_importance_gini'] = (0.5 - lorenz_area) / 0.5 if lorenz_area <=0.5 else 0.0 # Ensure Gini is between 0 and 1


        # --- Log Loss on provided data (if applicable) ---
        if y_true is not None and y_proba is not None:
            try:
                # Ensure y_true is in the format expected by log_loss (e.g., 0/1 for binary)
                # If self.label_encoder_ was used, y_true might need transformation
                # For simplicity, assuming y_true is already appropriately encoded if y_proba is given
                metrics['log_loss_on_provided_data'] = log_loss(y_true, y_proba, labels=self.model_.classes_)
            except ValueError as e:
                metrics['log_loss_on_provided_data_error'] = str(e)
            except Exception: # pylint: disable=broad-except
                metrics['log_loss_on_provided_data_error'] = "Could not compute"
        
        # Remove NaN or None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if pd.notna(v)}

        return metrics

def get_plugin():
    """Factory function to get plugin instance"""
    return GradientBoostingClassifierPlugin()
