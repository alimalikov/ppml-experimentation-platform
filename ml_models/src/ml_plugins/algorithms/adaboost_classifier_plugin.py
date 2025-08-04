import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Try to import AdaBoost with graceful fallback
try:
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    AdaBoostClassifier = None
    DecisionTreeClassifier = None

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

class AdaBoostClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    AdaBoost Classifier Plugin - Adaptive Boosting Pioneer
    
    AdaBoost (Adaptive Boosting) is the original and most famous boosting algorithm,
    developed by Freund and Schapire. It adaptively adjusts to focus on misclassified
    examples by increasing their weights, creating a strong classifier from weak learners.
    """
    
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 algorithm='SAMME.R',
                 random_state=42,
                 # Base estimator parameters
                 base_estimator_type='decision_tree',
                 max_depth=1,  # For decision stumps
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 # Advanced parameters
                 sample_weight_seed=None,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=10,
                 tol=1e-4):
        """
        Initialize AdaBoost Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        base_estimator : object, optional
            Base learner (default: DecisionTreeClassifier with max_depth=1)
        n_estimators : int, default=50
            Number of weak learners to train sequentially
        learning_rate : float, default=1.0
            Learning rate shrinks contribution of each classifier
        algorithm : str, default='SAMME.R'
            Boosting algorithm ('SAMME', 'SAMME.R')
        random_state : int, default=42
            Random seed for reproducibility
        base_estimator_type : str, default='decision_tree'
            Type of base estimator ('decision_tree', 'logistic', 'svm')
        max_depth : int, default=1
            Maximum depth of decision tree base learners (1 = stumps)
        min_samples_split : int, default=2
            Minimum samples required to split internal node
        min_samples_leaf : int, default=1
            Minimum samples required at leaf node
        max_features : str/int/float, optional
            Number of features to consider for best split
        early_stopping : bool, default=False
            Enable early stopping based on validation score
        validation_fraction : float, default=0.1
            Fraction of training data for validation (if early stopping)
        n_iter_no_change : int, default=10
            Number of iterations with no improvement to trigger early stopping
        tol : float, default=1e-4
            Tolerance for early stopping improvement threshold
        """
        super().__init__()
        
        # Core AdaBoost parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        
        # Base estimator configuration
        self.base_estimator_type = base_estimator_type
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
        # Advanced parameters
        self.sample_weight_seed = sample_weight_seed
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        
        # Plugin metadata
        self._name = "AdaBoost"
        self._description = "Adaptive Boosting - The original boosting algorithm that adaptively focuses on misclassified examples."
        self._category = "Tree-Based Models"
        self._algorithm_type = "Adaptive Boosting Ensemble"
        self._paper_reference = "Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 20
        self._handles_missing_values = False
        self._requires_scaling = False
        self._supports_sparse = False
        self._is_linear = False
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = False
        self._ensemble_method = True
        self._supports_early_stopping = True
        self._interpretable = True  # Via weak learners
        self._classical_algorithm = True
        self._foundational_method = True
        self._adaptive_weighting = True
        
        # Internal attributes
        self.model_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.training_history_ = None
        self.feature_importances_history_ = None
        self.estimator_weights_ = None
        self.estimator_errors_ = None
        
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
            "year_introduced": 1995,
            "key_innovations": {
                "adaptive_weighting": "Increases weights of misclassified examples",
                "weak_learner_combination": "Combines weak learners into strong classifier",
                "theoretical_guarantees": "Provable bounds on generalization error",
                "exponential_loss": "Minimizes exponential loss function",
                "forward_stagewise": "Adds one weak learner at a time",
                "voting_weights": "Weighted voting based on individual accuracy"
            },
            "algorithm_mechanics": {
                "boosting_process": {
                    "initialization": "Equal weights for all training examples",
                    "iteration": "Train weak learner on weighted dataset",
                    "error_calculation": "Compute weighted error rate",
                    "weight_update": "Increase weights of misclassified examples",
                    "classifier_weight": "Weight classifier by its accuracy",
                    "final_prediction": "Weighted majority vote"
                },
                "weight_update_formula": "w_i = w_i * exp(α * I(y_i ≠ h_t(x_i)))",
                "classifier_weight": "α_t = 0.5 * ln((1 - ε_t) / ε_t)",
                "final_hypothesis": "H(x) = sign(Σ α_t * h_t(x))"
            },
            "strengths": [
                "Simple and easy to understand",
                "Works well with weak learners",
                "Reduces both bias and variance",
                "Strong theoretical foundation",
                "Adaptive to difficult examples",
                "No need to know weak learner performance beforehand",
                "Can achieve arbitrary accuracy with enough weak learners",
                "Combines multiple simple models effectively",
                "Good performance on many datasets",
                "Interpretable through weak learner analysis",
                "Fast training with simple base learners",
                "Handles feature interactions through sequential learning"
            ],
            "weaknesses": [
                "Sensitive to noise and outliers",
                "Can overfit with too many estimators",
                "Performance depends heavily on base learner choice",
                "Vulnerable to uniform noise",
                "Sequential training (not parallelizable)",
                "May struggle with very complex patterns",
                "Exponential loss can be unstable",
                "Requires careful hyperparameter tuning",
                "Can be slow with complex base learners",
                "May not work well with very weak learners"
            ],
            "ideal_use_cases": [
                "Binary classification problems",
                "Datasets with clear patterns but some noise",
                "When interpretability is important",
                "Combining multiple simple models",
                "Face detection and recognition",
                "Text classification",
                "Object recognition",
                "Medical diagnosis",
                "Fraud detection",
                "Educational applications (concept learning)",
                "Baseline ensemble method",
                "When you have good weak learners available"
            ],
            "base_learner_options": {
                "decision_stumps": {
                    "description": "Single-level decision trees (most common)",
                    "advantages": ["Fast", "Interpretable", "Handles interactions"],
                    "parameters": ["max_depth=1", "min_samples_leaf", "max_features"]
                },
                "shallow_trees": {
                    "description": "Decision trees with limited depth",
                    "advantages": ["More expressive", "Better feature interactions"],
                    "parameters": ["max_depth=2-5", "min_samples_split", "pruning"]
                },
                "linear_models": {
                    "description": "Logistic regression or linear SVM",
                    "advantages": ["Fast", "Simple", "Good for linearly separable data"],
                    "parameters": ["regularization", "solver", "tolerance"]
                },
                "neural_networks": {
                    "description": "Simple neural networks",
                    "advantages": ["Non-linear", "Flexible"],
                    "disadvantages": ["Slower", "More complex"]
                }
            },
            "algorithm_variants": {
                "discrete_adaboost": {
                    "algorithm": "SAMME",
                    "description": "Original discrete AdaBoost",
                    "output": "Class predictions only",
                    "requirements": "Base learners must beat random guessing"
                },
                "real_adaboost": {
                    "algorithm": "SAMME.R",
                    "description": "Real AdaBoost with probability estimates",
                    "output": "Class probability estimates",
                    "advantages": ["Faster convergence", "Better performance"],
                    "requirements": "Base learners must support predict_proba"
                }
            },
            "hyperparameter_guide": {
                "n_estimators": "Start with 50, increase to 100-500 if underfitting",
                "learning_rate": "1.0 for stable convergence, 0.1-0.5 for regularization",
                "base_estimator": "Decision stumps (max_depth=1) are most common",
                "algorithm": "SAMME.R usually better than SAMME",
                "max_depth": "1 for stumps, 2-3 for shallow trees",
                "early_stopping": "Use with validation set to prevent overfitting"
            },
            "theoretical_properties": {
                "pac_learning": "Probably Approximately Correct learning guarantees",
                "generalization_bound": "Training error + complexity penalty",
                "bias_variance": "Reduces both bias and variance",
                "convergence": "Exponential convergence to minimum error",
                "margin_maximization": "Implicitly maximizes classification margin"
            },
            "comparison_with_other_methods": {
                "vs_bagging": {
                    "variance_reduction": "AdaBoost reduces bias, Bagging reduces variance",
                    "base_learner_requirement": "AdaBoost needs weak learners, Bagging works with strong learners",
                    "parallelization": "Bagging is parallelizable, AdaBoost is sequential",
                    "overfitting": "AdaBoost more prone to overfitting than Bagging"
                },
                "vs_gradient_boosting": {
                    "loss_function": "AdaBoost uses exponential loss, GBM uses arbitrary losses",
                    "flexibility": "GBM more flexible in loss functions",
                    "simplicity": "AdaBoost simpler to understand and implement",
                    "performance": "GBM often better performance, AdaBoost more interpretable"
                },
                "vs_random_forest": {
                    "interpretability": "AdaBoost more interpretable through sequential learning",
                    "robustness": "Random Forest more robust to noise",
                    "speed": "Random Forest parallelizable, faster training",
                    "feature_importance": "Both provide feature importance but differently"
                }
            }
        }
    
    def _create_base_estimator(self):
        """Create base estimator based on configuration"""
        if self.base_estimator is not None:
            return self.base_estimator
        
        if self.base_estimator_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
        elif self.base_estimator_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=100,
                solver='liblinear'
            )
        elif self.base_estimator_type == 'svm':
            return SVC(
                kernel='linear',
                probability=True,  # Needed for SAMME.R
                random_state=self.random_state
            )
        else:
            # Default to decision stump
            return DecisionTreeClassifier(
                max_depth=1,
                random_state=self.random_state
            )
    
    def fit(self, X, y, 
            sample_weight=None,
            monitor_training=True):
        """
        Fit the AdaBoost Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, optional
            Sample weights
        monitor_training : bool, default=True
            Whether to monitor training progress
            
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
        
        # Create base estimator
        base_estimator = self._create_base_estimator()
        
        # Determine algorithm based on base estimator capabilities
        algorithm = self.algorithm
        if algorithm == 'SAMME.R':
            # Check if base estimator supports predict_proba
            if not hasattr(base_estimator, 'predict_proba'):
                algorithm = 'SAMME'
                warnings.warn("Base estimator doesn't support predict_proba. Using SAMME algorithm.")
        
        # Create AdaBoost model
        self.model_ = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=algorithm,
            random_state=self.random_state
        )
        
        # Train the model
        if monitor_training:
            # Train with staged predictions to monitor progress
            self.training_history_ = {
                'train_errors': [],
                'estimator_weights': [],
                'estimator_errors': []
            }
            
            # Fit the model
            self.model_.fit(X, y_encoded, sample_weight=sample_weight)
            
            # Store training information
            if hasattr(self.model_, 'estimator_weights_'):
                self.estimator_weights_ = self.model_.estimator_weights_
            if hasattr(self.model_, 'estimator_errors_'):
                self.estimator_errors_ = self.model_.estimator_errors_
            
            # Calculate training error progression
            for i, pred in enumerate(self.model_.staged_predict(X)):
                error = np.mean(pred != y_encoded)
                self.training_history_['train_errors'].append(error)
        else:
            # Simple fit
            self.model_.fit(X, y_encoded, sample_weight=sample_weight)
        
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
        
        # Get predictions
        y_pred_encoded = self.model_.predict(X)
        
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
        
        # Get probability predictions
        probabilities = self.model_.predict_proba(X)
        
        return probabilities
    
    def decision_function(self, X):
        """
        Compute the decision function of X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        decision : array, shape (n_samples,) or (n_samples, n_classes)
            Decision function values
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        return self.model_.decision_function(X)
    
    def staged_predict(self, X):
        """
        Return staged predictions for X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Yields:
        -------
        y_pred : array, shape (n_samples,)
            Predictions after each stage
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        for y_pred_encoded in self.model_.staged_predict(X):
            yield self.label_encoder_.inverse_transform(y_pred_encoded)
    
    def staged_predict_proba(self, X):
        """
        Return staged class probabilities for X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Yields:
        -------
        probabilities : array, shape (n_samples, n_classes)
            Class probabilities after each stage
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        for probabilities in self.model_.staged_predict_proba(X):
            yield probabilities
    
    def get_feature_importance(self):
        """
        Get feature importance from the ensemble
        
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores
        """
        if not self.is_fitted_:
            return None
        
        return self.model_.feature_importances_
    
    def get_boosting_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of the boosting process
        
        Returns:
        --------
        analysis_info : dict
            Information about the boosting process
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "algorithm_used": self.model_.algorithm,
            "base_estimator_type": str(type(self.model_.base_estimator_)).split('.')[-1].replace("'>", ""),
            "n_estimators_trained": len(self.model_.estimators_),
            "learning_rate": self.learning_rate,
            "boosting_mechanics": {
                "sequential_learning": True,
                "adaptive_weighting": True,
                "weak_learner_combination": True,
                "exponential_loss_minimization": True
            }
        }
        
        # Add estimator weights and errors if available
        if hasattr(self.model_, 'estimator_weights_') and self.model_.estimator_weights_ is not None:
            analysis["estimator_weights"] = {
                "weights": self.model_.estimator_weights_.tolist(),
                "mean_weight": float(np.mean(self.model_.estimator_weights_)),
                "std_weight": float(np.std(self.model_.estimator_weights_)),
                "max_weight": float(np.max(self.model_.estimator_weights_)),
                "min_weight": float(np.min(self.model_.estimator_weights_))
            }
        
        if hasattr(self.model_, 'estimator_errors_') and self.model_.estimator_errors_ is not None:
            analysis["estimator_errors"] = {
                "errors": self.model_.estimator_errors_.tolist(),
                "mean_error": float(np.mean(self.model_.estimator_errors_)),
                "std_error": float(np.std(self.model_.estimator_errors_)),
                "final_error": float(self.model_.estimator_errors_[-1]) if len(self.model_.estimator_errors_) > 0 else None
            }
        
        # Add training history if available
        if self.training_history_:
            analysis["training_progression"] = self.training_history_
        
        return analysis
    
    def plot_boosting_analysis(self, figsize=(15, 10)):
        """
        Create comprehensive boosting analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Boosting analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Training Error Progression
        if self.training_history_ and self.training_history_['train_errors']:
            iterations = range(1, len(self.training_history_['train_errors']) + 1)
            ax1.plot(iterations, self.training_history_['train_errors'], 'b-', linewidth=2, marker='o', markersize=3)
            ax1.set_xlabel('Boosting Iteration')
            ax1.set_ylabel('Training Error Rate')
            ax1.set_title('AdaBoost Training Error Progression')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(self.training_history_['train_errors']) * 1.1)
        else:
            ax1.text(0.5, 0.5, 'Training history\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Training Error Progression')
        
        # 2. Estimator Weights
        if hasattr(self.model_, 'estimator_weights_') and self.model_.estimator_weights_ is not None:
            estimator_indices = range(len(self.model_.estimator_weights_))
            bars = ax2.bar(estimator_indices, self.model_.estimator_weights_, 
                          color='green', alpha=0.7, edgecolor='darkgreen')
            ax2.set_xlabel('Estimator Index')
            ax2.set_ylabel('Estimator Weight (α)')
            ax2.set_title('AdaBoost Estimator Weights')
            ax2.grid(True, alpha=0.3)
            
            # Add mean line
            mean_weight = np.mean(self.model_.estimator_weights_)
            ax2.axhline(y=mean_weight, color='red', linestyle='--', 
                       label=f'Mean: {mean_weight:.3f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Estimator weights\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Estimator Weights')
        
        # 3. Estimator Errors
        if hasattr(self.model_, 'estimator_errors_') and self.model_.estimator_errors_ is not None:
            estimator_indices = range(len(self.model_.estimator_errors_))
            ax3.plot(estimator_indices, self.model_.estimator_errors_, 'ro-', 
                    linewidth=2, markersize=4, alpha=0.7)
            ax3.set_xlabel('Estimator Index')
            ax3.set_ylabel('Weighted Error Rate')
            ax3.set_title('Individual Estimator Error Rates')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, 
                       label='Random Guessing (0.5)')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Estimator errors\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Estimator Error Rates')
        
        # 4. Feature Importance
        importance = self.get_feature_importance()
        if importance is not None:
            # Show top 15 features
            top_indices = np.argsort(importance)[::-1][:15]
            top_features = [self.feature_names_[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            bars = ax4.barh(range(len(top_features)), top_importance, 
                           color='orange', alpha=0.7, edgecolor='darkorange')
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in top_features])
            ax4.invert_yaxis()
            ax4.set_xlabel('Feature Importance')
            ax4.set_title(f'Top {len(top_features)} Feature Importances')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width + max(top_importance) * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, X, y, figsize=(12, 8)):
        """
        Plot learning curves showing performance vs number of estimators
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training targets
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Learning curves plot
        """
        if not self.is_fitted_:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Encode labels
        y_encoded = self.label_encoder_.transform(y)
        
        # Get staged predictions
        train_errors = []
        train_scores = []
        
        for pred in self.model_.staged_predict(X):
            error = np.mean(pred != y_encoded)
            score = 1 - error
            train_errors.append(error)
            train_scores.append(score)
        
        iterations = range(1, len(train_errors) + 1)
        
        # Plot 1: Error Rate
        ax1.plot(iterations, train_errors, 'b-', linewidth=2, label='Training Error', marker='o', markersize=3)
        ax1.set_xlabel('Number of Estimators')
        ax1.set_ylabel('Error Rate')
        ax1.set_title('AdaBoost Learning Curve - Error Rate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, max(train_errors) * 1.1)
        
        # Plot 2: Accuracy Score
        ax2.plot(iterations, train_scores, 'g-', linewidth=2, label='Training Accuracy', marker='s', markersize=3)
        ax2.set_xlabel('Number of Estimators')
        ax2.set_ylabel('Accuracy Score')
        ax2.set_title('AdaBoost Learning Curve - Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(min(train_scores) * 0.95, 1.05)
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🚀 AdaBoost Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Base Estimator", "Advanced", "Analysis", "Info"])
        
        with tab1:
            st.markdown("**Core AdaBoost Parameters**")
            
            # Number of estimators
            n_estimators = st.slider(
                "Number of Estimators:",
                min_value=10,
                max_value=500,
                value=int(self.n_estimators),
                step=10,
                help="Number of weak learners to train sequentially",
                key=f"{key_prefix}_n_estimators"
            )
            
            # Learning rate
            learning_rate = st.number_input(
                "Learning Rate:",
                value=float(self.learning_rate),
                min_value=0.01,
                max_value=2.0,
                step=0.1,
                help="Shrinkage rate - reduces contribution of each classifier",
                key=f"{key_prefix}_learning_rate"
            )
            
            # Algorithm
            algorithm = st.selectbox(
                "Boosting Algorithm:",
                options=['SAMME.R', 'SAMME'],
                index=['SAMME.R', 'SAMME'].index(self.algorithm),
                help="SAMME.R: uses probability estimates (faster), SAMME: uses class predictions",
                key=f"{key_prefix}_algorithm"
            )
            
            # Random state
            random_state = st.number_input(
                "Random Seed:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
        
        with tab2:
            st.markdown("**Base Estimator Configuration**")
            
            # Base estimator type
            base_estimator_type = st.selectbox(
                "Base Estimator Type:",
                options=['decision_tree', 'logistic', 'svm'],
                index=['decision_tree', 'logistic', 'svm'].index(self.base_estimator_type),
                help="Type of weak learner to use",
                key=f"{key_prefix}_base_estimator_type"
            )
            
            if base_estimator_type == 'decision_tree':
                st.markdown("**Decision Tree Parameters**")
                
                # Max depth
                max_depth = st.slider(
                    "Max Depth:",
                    min_value=1,
                    max_value=10,
                    value=int(self.max_depth),
                    help="1 = decision stumps (recommended), higher = more complex trees",
                    key=f"{key_prefix}_max_depth"
                )
                
                # Min samples split
                min_samples_split = st.slider(
                    "Min Samples Split:",
                    min_value=2,
                    max_value=20,
                    value=int(self.min_samples_split),
                    help="Minimum samples required to split internal node",
                    key=f"{key_prefix}_min_samples_split"
                )
                
                # Min samples leaf
                min_samples_leaf = st.slider(
                    "Min Samples Leaf:",
                    min_value=1,
                    max_value=20,
                    value=int(self.min_samples_leaf),
                    help="Minimum samples required at leaf node",
                    key=f"{key_prefix}_min_samples_leaf"
                )
                
                # Max features
                max_features_option = st.selectbox(
                    "Max Features:",
                    options=['None', 'sqrt', 'log2', 'auto'],
                    index=0,
                    help="Number of features to consider for best split",
                    key=f"{key_prefix}_max_features_option"
                )
                
                max_features = None if max_features_option == 'None' else max_features_option
                
            else:
                # For non-tree estimators, use defaults
                max_depth = 1
                min_samples_split = 2
                min_samples_leaf = 1
                max_features = None
        
        with tab3:
            st.markdown("**Advanced Settings**")
            
            # Early stopping
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=self.early_stopping,
                help="Stop training early if no improvement",
                key=f"{key_prefix}_early_stopping"
            )
            
            if early_stopping:
                validation_fraction = st.slider(
                    "Validation Fraction:",
                    min_value=0.05,
                    max_value=0.3,
                    value=float(self.validation_fraction),
                    step=0.05,
                    help="Fraction of training data for validation",
                    key=f"{key_prefix}_validation_fraction"
                )
                
                n_iter_no_change = st.slider(
                    "Patience (iterations):",
                    min_value=5,
                    max_value=50,
                    value=int(self.n_iter_no_change),
                    help="Number of iterations with no improvement",
                    key=f"{key_prefix}_n_iter_no_change"
                )
                
                tol = st.number_input(
                    "Tolerance:",
                    value=float(self.tol),
                    min_value=1e-6,
                    max_value=1e-2,
                    step=1e-4,
                    format="%.6f",
                    help="Minimum improvement threshold",
                    key=f"{key_prefix}_tol"
                )
            else:
                validation_fraction = self.validation_fraction
                n_iter_no_change = self.n_iter_no_change
                tol = self.tol
        
        with tab4:
            st.markdown("**Training Analysis**")
            
            # Training monitoring
            monitor_training = st.checkbox(
                "Monitor Training Progress",
                value=True,
                help="Track training errors and estimator weights",
                key=f"{key_prefix}_monitor_training"
            )
            
            # Visualization options
            st.markdown("**Post-Training Visualizations:**")
            show_boosting_analysis = st.checkbox(
                "Boosting Analysis Plots",
                value=True,
                help="Show estimator weights, errors, and feature importance",
                key=f"{key_prefix}_show_boosting_analysis"
            )
            
            show_learning_curves = st.checkbox(
                "Learning Curves",
                value=True,
                help="Show performance vs number of estimators",
                key=f"{key_prefix}_show_learning_curves"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if SKLEARN_AVAILABLE:
                st.success("✅ scikit-learn is available")
            else:
                st.error("❌ scikit-learn not installed. Run: pip install scikit-learn")
            
            st.info("""
            **AdaBoost** - Adaptive Boosting Pioneer:
            • 🚀 Original boosting algorithm (1995)
            • 🎯 Adaptively focuses on difficult examples
            • 🧠 Combines weak learners into strong classifier
            • 📊 Theoretical guarantees on performance
            • 🔍 Highly interpretable through weak learners
            • ⚡ Works well with decision stumps
            
            **Key Advantages:**
            • Simple and intuitive algorithm
            • Strong theoretical foundation
            • Excellent with decision stumps
            • Reduces both bias and variance
            """)
            
            # Algorithm explanation
            if st.button("🧮 How AdaBoost Works", key=f"{key_prefix}_how_it_works"):
                st.markdown("""
                **AdaBoost Algorithm Steps:**
                
                1. **Initialize:** Equal weights for all examples
                2. **For each iteration:**
                   - Train weak learner on weighted data
                   - Calculate weighted error rate
                   - Compute classifier importance (α)
                   - Update example weights (increase for misclassified)
                3. **Final prediction:** Weighted vote of all classifiers
                
                **Key Formula:**
                - α_t = 0.5 * ln((1 - ε_t) / ε_t)
                - w_i ← w_i * exp(α_t * I(y_i ≠ h_t(x_i)))
                """)
            
            # Base estimator guide
            if st.button("🌳 Base Estimator Guide", key=f"{key_prefix}_base_guide"):
                st.markdown("""
                **Choosing Base Estimators:**
                
                **Decision Stumps (max_depth=1):**
                - Most common and effective
                - Fast training and prediction
                - Good interpretability
                - Handles feature interactions through boosting
                
                **Shallow Trees (max_depth=2-3):**
                - More expressive than stumps
                - Can capture some interactions directly
                - Risk of overfitting with deep trees
                
                **Linear Models:**
                - Good for linearly separable data
                - Fast and simple
                - Works well with SAMME.R algorithm
                """)
            
            # Hyperparameter tuning guide
            if st.button("🎯 Tuning Strategy", key=f"{key_prefix}_tuning_strategy"):
                st.markdown("""
                **AdaBoost Tuning Strategy:**
                
                **Step 1: Start Simple**
                - Use decision stumps (max_depth=1)
                - Start with 50-100 estimators
                - Use learning_rate=1.0
                - Use SAMME.R algorithm
                
                **Step 2: Adjust Core Parameters**
                - Increase n_estimators if underfitting
                - Reduce learning_rate if overfitting
                - Try SAMME if SAMME.R doesn't work
                
                **Step 3: Base Estimator Tuning**
                - Experiment with max_depth=2-3
                - Adjust min_samples_leaf for regularization
                - Try different base estimator types
                
                **Step 4: Advanced Options**
                - Use early stopping for automatic tuning
                - Monitor training curves for optimal stopping
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "algorithm": algorithm,
            "random_state": random_state,
            "base_estimator_type": base_estimator_type,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "early_stopping": early_stopping,
            "validation_fraction": validation_fraction,
            "n_iter_no_change": n_iter_no_change,
            "tol": tol,
            "_ui_options": {
                "monitor_training": monitor_training,
                "show_boosting_analysis": show_boosting_analysis,
                "show_learning_curves": show_learning_curves
            }
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return AdaBoostClassifierPlugin(
            n_estimators=hyperparameters.get("n_estimators", self.n_estimators),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            algorithm=hyperparameters.get("algorithm", self.algorithm),
            random_state=hyperparameters.get("random_state", self.random_state),
            base_estimator_type=hyperparameters.get("base_estimator_type", self.base_estimator_type),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            min_samples_split=hyperparameters.get("min_samples_split", self.min_samples_split),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", self.min_samples_leaf),
            max_features=hyperparameters.get("max_features", self.max_features),
            early_stopping=hyperparameters.get("early_stopping", self.early_stopping),
            validation_fraction=hyperparameters.get("validation_fraction", self.validation_fraction),
            n_iter_no_change=hyperparameters.get("n_iter_no_change", self.n_iter_no_change),
            tol=hyperparameters.get("tol", self.tol)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for AdaBoost
        
        AdaBoost works best with clean, preprocessed data.
        Missing values should be handled before training.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # AdaBoost doesn't handle missing values natively
        if np.any(pd.isna(X_processed)):
            warnings.warn("AdaBoost doesn't handle missing values well. Consider imputation.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if AdaBoost is compatible with the given data
        
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
            return False, f"AdaBoost requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for missing values
        if np.any(pd.isna(X)):
            return False, "AdaBoost doesn't handle missing values well. Please impute missing values first."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            # AdaBoost works best with balanced classes
            if len(unique_values) > 10:
                return True, "AdaBoost works with multi-class problems but may be slower with many classes"
        
        return True, "AdaBoost is compatible with this data"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "algorithm": self.model_.algorithm,
            "base_estimator": str(type(self.model_.base_estimator_)).split('.')[-1].replace("'>", ""),
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "n_estimators_trained": len(self.model_.estimators_),
            "final_estimator_weight": self.model_.estimator_weights_[-1] if hasattr(self.model_, 'estimator_weights_') and len(self.model_.estimator_weights_) > 0 else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "AdaBoost",
            "training_completed": True,
            "boosting_characteristics": {
                "adaptive_weighting": True,
                "sequential_learning": True,
                "weak_learner_combination": True,
                "exponential_loss": True,
                "theoretical_guarantees": True
            },
            "model_details": {
                "algorithm_variant": self.model_.algorithm,
                "base_estimator_type": str(type(self.model_.base_estimator_)).split('.')[-1].replace("'>", ""),
                "n_estimators_trained": len(self.model_.estimators_),
                "learning_rate": self.learning_rate
            },
            "boosting_analysis": self.get_boosting_analysis()
        }
        
        return info

    def get_algorithm_specific_metrics(self, 
                                       y_true: Union[pd.Series, np.ndarray], 
                                       y_pred: Union[pd.Series, np.ndarray], 
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate AdaBoost-specific metrics related to its training and ensemble structure.

        Args:
            y_true: Ground truth target values from the test set.
            y_pred: Predicted target values on the test set.
            y_proba: Predicted probabilities on the test set (for classification tasks), if available.

        Returns:
            A dictionary of AdaBoost-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            return {"status": "Model not fitted or not available"}

        # Metrics from the boosting process (internal model attributes)
        if hasattr(self.model_, 'estimator_weights_') and self.model_.estimator_weights_ is not None:
            metrics['mean_estimator_weight'] = float(np.mean(self.model_.estimator_weights_))
            metrics['std_estimator_weight'] = float(np.std(self.model_.estimator_weights_))
            metrics['min_estimator_weight'] = float(np.min(self.model_.estimator_weights_))
            metrics['max_estimator_weight'] = float(np.max(self.model_.estimator_weights_))
        
        if hasattr(self.model_, 'estimator_errors_') and self.model_.estimator_errors_ is not None:
            metrics['mean_estimator_error_rate'] = float(np.mean(self.model_.estimator_errors_))
            metrics['std_estimator_error_rate'] = float(np.std(self.model_.estimator_errors_))
            if len(self.model_.estimator_errors_) > 0:
                metrics['final_estimator_error_rate'] = float(self.model_.estimator_errors_[-1])
        
        if hasattr(self.model_, 'estimators_'):
            metrics['n_estimators_trained'] = len(self.model_.estimators_)

        # Feature importances (if available and considered specific enough)
        if hasattr(self.model_, 'feature_importances_') and self.model_.feature_importances_ is not None:
            # Storing all feature importances might be too verbose for a simple metric display.
            # Consider storing top N or a summary if needed, or let feature importance plots handle this.
            # For now, let's add the mean importance as an example.
            metrics['mean_feature_importance'] = float(np.mean(self.model_.feature_importances_))

        # Training progression metrics (if training history was monitored)
        if self.training_history_ and 'train_errors' in self.training_history_ and self.training_history_['train_errors']:
            metrics['final_training_error_rate_on_ensemble'] = float(self.training_history_['train_errors'][-1])
            metrics['min_training_error_rate_on_ensemble'] = float(min(self.training_history_['train_errors']))
        
        # Example of using y_proba if available (e.g., for log_loss, though it's a common metric)
        # if y_proba is not None and self._supports_classification:
        #     try:
        #         from sklearn.metrics import log_loss
        #         # Ensure y_true is in the same format as classes for log_loss
        #         # This might require label encoding if y_true is not 0,1,...
        #         # For simplicity, assuming y_true is already suitable or handled by the caller
        #         metrics['log_loss_test_set'] = log_loss(y_true, y_proba, labels=self.model_.classes_ if hasattr(self.model_, 'classes_') else None)
        #     except Exception as e:
        #         metrics['log_loss_test_set_error'] = str(e)
                
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return AdaBoostClassifierPlugin()
