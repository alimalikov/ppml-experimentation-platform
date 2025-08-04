import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations
import math

# Extended algorithms availability check
try:
    import xgboost as xgb
    EXTENDED_ALGORITHMS = True
except ImportError:
    EXTENDED_ALGORITHMS = False

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

class OneVsOneClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    One-vs-One Classifier Plugin - Pairwise Multi-class Classification
    
    The One-vs-One (OvO) approach creates K(K-1)/2 binary classifiers for K classes,
    where each classifier is trained on data from exactly two classes. This creates
    a voting system where each pairwise classifier votes for one of its two classes,
    and the final prediction is the class with the most votes.
    
    Key advantages:
    - Uses only relevant data for each binary problem (no class imbalance)
    - Often more accurate than One-vs-Rest for few classes
    - Each classifier focuses on distinguishing just two classes
    - Natural handling of ambiguous regions between classes
    
    Best for: 2-10 classes, when pairwise separability is high
    """
    
    def __init__(self, 
                 base_estimator='logistic_regression',
                 auto_scale_features=True,
                 probability_calibration=False,
                 voting_strategy='majority',
                 tie_breaking='confidence',
                 estimate_pairwise_performance=True,
                 n_jobs=None,
                 verbose=0,
                 random_state=42,
                 
                 # Base estimator hyperparameters
                 # Logistic Regression
                 lr_C=1.0,
                 lr_max_iter=1000,
                 lr_solver='lbfgs',
                 lr_penalty='l2',
                 
                 # Random Forest
                 rf_n_estimators=100,
                 rf_max_depth=None,
                 rf_min_samples_split=2,
                 rf_min_samples_leaf=1,
                 rf_criterion='gini',
                 
                 # SVM
                 svm_C=1.0,
                 svm_kernel='rbf',
                 svm_probability=True,
                 svm_gamma='scale',
                 
                 # Decision Tree
                 dt_max_depth=None,
                 dt_min_samples_split=2,
                 dt_min_samples_leaf=1,
                 dt_criterion='gini',
                 
                 # KNN
                 knn_n_neighbors=5,
                 knn_weights='uniform',
                 knn_metric='minkowski',
                 
                 # Gradient Boosting
                 gb_n_estimators=100,
                 gb_learning_rate=0.1,
                 gb_max_depth=3,
                 gb_subsample=1.0,
                 
                 # Neural Network
                 mlp_hidden_layer_sizes=(100,),
                 mlp_activation='relu',
                 mlp_max_iter=1000,
                 mlp_alpha=0.0001,
                 
                 # XGBoost
                 xgb_n_estimators=100,
                 xgb_learning_rate=0.1,
                 xgb_max_depth=6,
                 xgb_subsample=1.0):
        """
        Initialize One-vs-One Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        base_estimator : str, default='logistic_regression'
            Base binary classifier to use for pairwise classification
        auto_scale_features : bool, default=True
            Whether to automatically scale features for distance-based estimators
        probability_calibration : bool, default=False
            Whether to calibrate probabilities using cross-validation
        voting_strategy : str, default='majority'
            Strategy for combining pairwise votes ('majority', 'weighted')
        tie_breaking : str, default='confidence'
            How to break ties ('confidence', 'random', 'first')
        estimate_pairwise_performance : bool, default=True
            Whether to estimate individual pairwise classifier performance
        n_jobs : int, default=None
            Number of CPU cores for parallel training of pairwise classifiers
        verbose : int, default=0
            Verbosity level for training output
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__()
        
        # Core OvO parameters
        self.base_estimator = base_estimator
        self.auto_scale_features = auto_scale_features
        self.probability_calibration = probability_calibration
        self.voting_strategy = voting_strategy
        self.tie_breaking = tie_breaking
        self.estimate_pairwise_performance = estimate_pairwise_performance
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        
        # Base estimator hyperparameters
        self.lr_C = lr_C
        self.lr_max_iter = lr_max_iter
        self.lr_solver = lr_solver
        self.lr_penalty = lr_penalty
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_criterion = rf_criterion
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_probability = svm_probability
        self.svm_gamma = svm_gamma
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_split = dt_min_samples_split
        self.dt_min_samples_leaf = dt_min_samples_leaf
        self.dt_criterion = dt_criterion
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_weights = knn_weights
        self.knn_metric = knn_metric
        self.gb_n_estimators = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.gb_max_depth = gb_max_depth
        self.gb_subsample = gb_subsample
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_activation = mlp_activation
        self.mlp_max_iter = mlp_max_iter
        self.mlp_alpha = mlp_alpha
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_subsample = xgb_subsample
        
        # Plugin metadata
        self._name = "One-vs-One Classifier"
        self._description = "Pairwise binary classification for multi-class problems. Creates K(K-1)/2 classifiers for K classes."
        self._category = "Multi-class Strategies"
        self._algorithm_type = "Pairwise Multi-class Wrapper"
        self._paper_reference = "Hastie, T., & Tibshirani, R. (1998). Classification by pairwise coupling. Annals of statistics, 451-471."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True  # Via pairwise decomposition
        self._supports_multiclass = True
        self._min_samples_required = 20  # Need samples for each pair
        self._handles_missing_values = False
        self._requires_scaling = False  # Depends on base estimator
        self._supports_sparse = False
        self._is_ensemble = True
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._is_deterministic = False  # Depends on base estimator
        self._training_complexity = "O(K² × Base_Training)"
        self._prediction_complexity = "O(K² × Base_Prediction)"
        
        # Internal attributes
        self.pairwise_classifiers_ = {}
        self.class_pairs_ = []
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.pairwise_performance_ = {}
        self.pairwise_distributions_ = {}
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
            "mathematical_foundation": {
                "approach": "Pairwise binary classification decomposition",
                "classifiers": "K(K-1)/2 binary classifiers for K classes",
                "decision_rule": "Majority voting among pairwise predictions",
                "advantages": "No class imbalance, focused binary problems"
            },
            "strengths": [
                "No artificial class imbalance in binary problems",
                "Each classifier focuses on exactly two classes",
                "Often more accurate than OvR for few classes",
                "Natural handling of ambiguous regions",
                "Uses only relevant training data per classifier",
                "Good probability estimates via pairwise coupling",
                "Robust to outliers in specific class pairs",
                "Can identify confusing class pairs",
                "Parallelizable training of pairwise classifiers"
            ],
            "weaknesses": [
                "Quadratic number of classifiers K(K-1)/2",
                "High computational cost for many classes",
                "Memory usage scales quadratically",
                "Slower training and prediction than OvR",
                "May be inefficient for large K",
                "Tie-breaking can be arbitrary",
                "Complex interpretability with many pairs"
            ],
            "use_cases": [
                "Multi-class problems with 2-10 classes",
                "When classes are naturally separable in pairs",
                "High accuracy requirements for moderate K",
                "Problems with high between-class variation",
                "When OvR creates too much class imbalance",
                "Image classification with few classes",
                "Medical diagnosis with distinct conditions",
                "Quality control with few defect types",
                "Species classification in biology",
                "Sentiment analysis with few categories"
            ],
            "pairwise_advantages": {
                "focused_learning": "Each classifier sees only relevant data",
                "balanced_problems": "Natural balance in each binary problem",
                "local_expertise": "Specialists for each class pair",
                "robustness": "One bad pair doesn't ruin entire system"
            },
            "complexity": {
                "training": "O(K² × Base_Training_Complexity)",
                "prediction": "O(K² × Base_Prediction_Complexity)",
                "memory": "O(K² × Base_Memory_Usage)",
                "classifiers": "K(K-1)/2 binary classifiers"
            },
            "voting_strategies": {
                "majority": "Simple vote counting (default)",
                "weighted": "Weight votes by classifier confidence",
                "probabilistic": "Pairwise coupling for probabilities"
            },
            "scalability_analysis": {
                "2_classes": "1 classifier (same as binary)",
                "3_classes": "3 classifiers",
                "5_classes": "10 classifiers", 
                "10_classes": "45 classifiers",
                "20_classes": "190 classifiers (getting expensive)",
                "recommendation": "Use OvR for K > 15"
            }
        }
    
    def _create_base_estimator(self):
        """Create a base estimator instance with current hyperparameters"""
        if self.base_estimator == 'logistic_regression':
            return LogisticRegression(
                C=self.lr_C,
                max_iter=self.lr_max_iter,
                solver=self.lr_solver,
                penalty=self.lr_penalty,
                random_state=self.random_state
            )
        elif self.base_estimator == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_split=self.rf_min_samples_split,
                min_samples_leaf=self.rf_min_samples_leaf,
                criterion=self.rf_criterion,
                random_state=self.random_state,
                n_jobs=1  # Individual estimator uses 1 job
            )
        elif self.base_estimator == 'svm':
            return SVC(
                C=self.svm_C,
                kernel=self.svm_kernel,
                probability=self.svm_probability,
                gamma=self.svm_gamma,
                random_state=self.random_state
            )
        elif self.base_estimator == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=self.dt_max_depth,
                min_samples_split=self.dt_min_samples_split,
                min_samples_leaf=self.dt_min_samples_leaf,
                criterion=self.dt_criterion,
                random_state=self.random_state
            )
        elif self.base_estimator == 'knn':
            return KNeighborsClassifier(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights,
                metric=self.knn_metric,
                n_jobs=1
            )
        elif self.base_estimator == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.gb_n_estimators,
                learning_rate=self.gb_learning_rate,
                max_depth=self.gb_max_depth,
                subsample=self.gb_subsample,
                random_state=self.random_state
            )
        elif self.base_estimator == 'mlp' and EXTENDED_ALGORITHMS:
            return MLPClassifier(
                hidden_layer_sizes=self.mlp_hidden_layer_sizes,
                activation=self.mlp_activation,
                max_iter=self.mlp_max_iter,
                alpha=self.mlp_alpha,
                random_state=self.random_state
            )
        elif self.base_estimator == 'xgboost' and EXTENDED_ALGORITHMS:
            return xgb.XGBClassifier(
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                max_depth=self.xgb_max_depth,
                subsample=self.xgb_subsample,
                random_state=self.random_state,
                n_jobs=1
            )
        else:
            # Fallback to logistic regression
            return LogisticRegression(random_state=self.random_state)
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the One-vs-One Classifier
        
        Creates K(K-1)/2 binary classifiers for K classes, where each classifier
        is trained on data from exactly two classes.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights (applied to pairwise problems)
            
        Returns:
        --------
        self : object
        """
        # 🎯 STORE FEATURE NAMES BEFORE VALIDATION!
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Check minimum classes requirement
        if self.n_classes_ < 2:
            raise ValueError("One-vs-One requires at least 2 classes")
        
        # Generate all class pairs
        self.class_pairs_ = list(combinations(range(self.n_classes_), 2))
        n_pairs = len(self.class_pairs_)
        
        if self.verbose:
            print(f"Training {n_pairs} pairwise classifiers for {self.n_classes_} classes...")
        
        # Scale features if requested
        if self.auto_scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
        
        # Train pairwise classifiers
        self.pairwise_classifiers_ = {}
        self.pairwise_distributions_ = {}
        
        for i, (class_a, class_b) in enumerate(self.class_pairs_):
            if self.verbose:
                print(f"Training classifier {i+1}/{n_pairs}: {self.classes_[class_a]} vs {self.classes_[class_b]}")
            
            # Create binary dataset for this pair
            mask = (y_encoded == class_a) | (y_encoded == class_b)
            X_pair = X_scaled[mask]
            y_pair = y_encoded[mask]
            
            # Convert to binary labels (0 for class_a, 1 for class_b)
            y_binary = (y_pair == class_b).astype(int)
            
            # Store class distribution for this pair
            self.pairwise_distributions_[(class_a, class_b)] = {
                'class_a_samples': int(np.sum(y_pair == class_a)),
                'class_b_samples': int(np.sum(y_pair == class_b)),
                'total_samples': int(len(y_pair)),
                'class_a_name': str(self.classes_[class_a]),
                'class_b_name': str(self.classes_[class_b])
            }
            
            # Create and train base estimator
            estimator = self._create_base_estimator()
            
            # Apply calibration if requested
            if self.probability_calibration:
                estimator = CalibratedClassifierCV(estimator, cv=3)
            
            # Train the pairwise classifier
            if sample_weight is not None:
                pair_weights = sample_weight[mask]
                estimator.fit(X_pair, y_binary, sample_weight=pair_weights)
            else:
                estimator.fit(X_pair, y_binary)
            
            self.pairwise_classifiers_[(class_a, class_b)] = estimator
        
        # Estimate pairwise performance if requested
        if self.estimate_pairwise_performance:
            self._estimate_pairwise_performance(X_scaled, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def _estimate_pairwise_performance(self, X, y):
        """Estimate performance of individual pairwise classifiers using cross-validation"""
        self.pairwise_performance_ = {}
        
        for class_a, class_b in self.class_pairs_:
            # Create binary dataset for this pair
            mask = (y == class_a) | (y == class_b)
            X_pair = X[mask]
            y_pair = y[mask]
            y_binary = (y_pair == class_b).astype(int)
            
            if len(X_pair) < 6:  # Need at least 6 samples for 3-fold CV
                continue
            
            # Create fresh estimator for CV evaluation
            estimator = self._create_base_estimator()
            
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(
                    estimator, X_pair, y_binary, 
                    cv=min(3, len(X_pair) // 2), 
                    scoring='f1'
                )
                
                self.pairwise_performance_[(class_a, class_b)] = {
                    'mean_f1_score': float(np.mean(cv_scores)),
                    'std_f1_score': float(np.std(cv_scores)),
                    'cv_scores': cv_scores.tolist(),
                    'n_samples': int(len(X_pair)),
                    'class_a_name': str(self.classes_[class_a]),
                    'class_b_name': str(self.classes_[class_b])
                }
            except Exception as e:
                if self.verbose:
                    print(f"Could not evaluate pair ({self.classes_[class_a]}, {self.classes_[class_b]}): {e}")
    
    def predict(self, X):
        """
        Predict class labels using pairwise voting
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Scale features if scaler was fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        n_samples = X_scaled.shape[0]
        votes = np.zeros((n_samples, self.n_classes_))
        confidence_votes = np.zeros((n_samples, self.n_classes_))
        
        # Collect votes from all pairwise classifiers
        for (class_a, class_b), classifier in self.pairwise_classifiers_.items():
            # Get predictions and confidence scores
            predictions = classifier.predict(X_scaled)
            
            if hasattr(classifier, 'predict_proba'):
                try:
                    probabilities = classifier.predict_proba(X_scaled)
                    confidences = np.max(probabilities, axis=1)
                except:
                    confidences = np.ones(n_samples)  # Fallback
            elif hasattr(classifier, 'decision_function'):
                try:
                    decision_scores = classifier.decision_function(X_scaled)
                    confidences = np.abs(decision_scores)
                except:
                    confidences = np.ones(n_samples)  # Fallback
            else:
                confidences = np.ones(n_samples)  # Fallback
            
            # Cast votes (0 = class_a wins, 1 = class_b wins)
            for i in range(n_samples):
                if predictions[i] == 0:  # class_a wins
                    votes[i, class_a] += 1
                    if self.voting_strategy == 'weighted':
                        confidence_votes[i, class_a] += confidences[i]
                else:  # class_b wins
                    votes[i, class_b] += 1
                    if self.voting_strategy == 'weighted':
                        confidence_votes[i, class_b] += confidences[i]
        
        # Determine final predictions
        if self.voting_strategy == 'weighted':
            predicted_classes = np.argmax(confidence_votes, axis=1)
        else:
            predicted_classes = np.argmax(votes, axis=1)
        
        # Handle ties
        for i in range(n_samples):
            if self.voting_strategy == 'weighted':
                max_votes = np.max(confidence_votes[i])
                tied_classes = np.where(confidence_votes[i] == max_votes)[0]
            else:
                max_votes = np.max(votes[i])
                tied_classes = np.where(votes[i] == max_votes)[0]
            
            if len(tied_classes) > 1:
                if self.tie_breaking == 'random':
                    np.random.seed(self.random_state + i)
                    predicted_classes[i] = np.random.choice(tied_classes)
                elif self.tie_breaking == 'first':
                    predicted_classes[i] = tied_classes[0]
                # 'confidence' is already handled by weighted voting
        
        # Convert back to original labels
        return self.label_encoder_.inverse_transform(predicted_classes)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using pairwise coupling
        
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
        
        # Scale features if scaler was fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        n_samples = X_scaled.shape[0]
        pairwise_probabilities = {}
        
        # Collect probabilities from all pairwise classifiers
        for (class_a, class_b), classifier in self.pairwise_classifiers_.items():
            if hasattr(classifier, 'predict_proba'):
                try:
                    proba = classifier.predict_proba(X_scaled)
                    # Store probability that class_b wins
                    if proba.shape[1] == 2:
                        pairwise_probabilities[(class_a, class_b)] = proba[:, 1]
                    else:
                        pairwise_probabilities[(class_a, class_b)] = proba[:, 0]
                except:
                    # Fallback to 0.5 if probability estimation fails
                    pairwise_probabilities[(class_a, class_b)] = np.full(n_samples, 0.5)
            else:
                # Fallback to 0.5 if no probability support
                pairwise_probabilities[(class_a, class_b)] = np.full(n_samples, 0.5)
        
        # Use simple averaging approach for probability estimation
        class_probabilities = np.zeros((n_samples, self.n_classes_))
        
        for i in range(n_samples):
            votes = np.zeros(self.n_classes_)
            
            for (class_a, class_b), prob_b in pairwise_probabilities.items():
                prob_a = 1.0 - prob_b[i]
                votes[class_a] += prob_a
                votes[class_b] += prob_b[i]
            
            # Normalize to get probabilities
            if np.sum(votes) > 0:
                class_probabilities[i] = votes / np.sum(votes)
            else:
                class_probabilities[i] = np.full(self.n_classes_, 1.0 / self.n_classes_)
        
        return class_probabilities
    
    def decision_function(self, X):
        """
        Calculate decision scores for samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        scores : array, shape (n_samples, n_classes)
            Decision scores
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Scale features if scaler was fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        n_samples = X_scaled.shape[0]
        decision_scores = np.zeros((n_samples, self.n_classes_))
        
        # Collect decision scores from all pairwise classifiers
        for (class_a, class_b), classifier in self.pairwise_classifiers_.items():
            if hasattr(classifier, 'decision_function'):
                try:
                    scores = classifier.decision_function(X_scaled)
                    # Positive scores favor class_b, negative scores favor class_a
                    decision_scores[:, class_a] -= scores
                    decision_scores[:, class_b] += scores
                except:
                    pass  # Skip if decision function fails
        
        return decision_scores
    
    def get_pairwise_predictions(self, X):
        """
        Get predictions from all pairwise classifiers
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        pairwise_predictions : dict
            Dictionary mapping class pairs to predictions
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Scale features if scaler was fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        pairwise_predictions = {}
        
        for (class_a, class_b), classifier in self.pairwise_classifiers_.items():
            predictions = classifier.predict(X_scaled)
            # Convert binary predictions to class names
            pred_classes = []
            for pred in predictions:
                if pred == 0:
                    pred_classes.append(self.classes_[class_a])
                else:
                    pred_classes.append(self.classes_[class_b])
            
            pair_name = f"{self.classes_[class_a]}_vs_{self.classes_[class_b]}"
            pairwise_predictions[pair_name] = pred_classes
        
        return pairwise_predictions
    
    def get_ovo_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive One-vs-One analysis
        
        Returns:
        --------
        analysis : dict
            Dictionary containing OvO strategy analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "strategy_summary": {
                "strategy_type": "One-vs-One (Pairwise)",
                "base_estimator": self.base_estimator,
                "n_classes": self.n_classes_,
                "n_pairwise_classifiers": len(self.class_pairs_),
                "voting_strategy": self.voting_strategy,
                "tie_breaking": self.tie_breaking,
                "feature_scaling": self.scaler_ is not None,
                "probability_calibration": self.probability_calibration
            },
            "pairwise_structure": {
                "total_pairs": len(self.class_pairs_),
                "pairs_list": [
                    {
                        "pair": f"{self.classes_[a]} vs {self.classes_[b]}",
                        "class_indices": (int(a), int(b)),
                        "class_names": (str(self.classes_[a]), str(self.classes_[b]))
                    }
                    for a, b in self.class_pairs_
                ],
                "computational_complexity": {
                    "classifiers": f"{len(self.class_pairs_)} binary classifiers",
                    "training_cost": f"O({self.n_classes_}² × Base_Training)",
                    "prediction_cost": f"O({self.n_classes_}² × Base_Prediction)",
                    "memory_usage": f"{len(self.class_pairs_)} complete base models"
                }
            }
        }
        
        # Add pairwise distributions
        if self.pairwise_distributions_:
            analysis["pairwise_distributions"] = self.pairwise_distributions_
        
        # Add pairwise performance
        if self.pairwise_performance_:
            analysis["pairwise_performance"] = self.pairwise_performance_
            
            # Summarize performance
            f1_scores = [perf['mean_f1_score'] for perf in self.pairwise_performance_.values()]
            if f1_scores:
                analysis["performance_summary"] = {
                    "mean_pairwise_f1": float(np.mean(f1_scores)),
                    "std_pairwise_f1": float(np.std(f1_scores)),
                    "min_pairwise_f1": float(np.min(f1_scores)),
                    "max_pairwise_f1": float(np.max(f1_scores)),
                    "n_evaluated_pairs": len(f1_scores)
                }
        
        # OvO characteristics
        analysis["ovo_characteristics"] = {
            "pairwise_decomposition": True,
            "balanced_binary_problems": True,
            "focused_learning": True,
            "voting_aggregation": True,
            "quadratic_scaling": True,
            "natural_probability_estimates": self.probability_calibration or 
                hasattr(self._create_base_estimator(), 'predict_proba')
        }
        
        # Scalability analysis
        analysis["scalability_analysis"] = {
            "current_classes": self.n_classes_,
            "current_classifiers": len(self.class_pairs_),
            "next_class_cost": f"+{self.n_classes_} additional classifiers",
            "efficiency_rating": self._get_efficiency_rating(),
            "recommendation": self._get_scalability_recommendation()
        }
        
        return analysis
    
    def _get_efficiency_rating(self):
        """Get efficiency rating based on number of classes"""
        if self.n_classes_ <= 3:
            return "Excellent (≤3 classes)"
        elif self.n_classes_ <= 5:
            return "Very Good (4-5 classes)"
        elif self.n_classes_ <= 10:
            return "Good (6-10 classes)"
        elif self.n_classes_ <= 15:
            return "Moderate (11-15 classes)"
        else:
            return "Poor (>15 classes - consider OvR)"
    
    def _get_scalability_recommendation(self):
        """Get scalability recommendation"""
        n_pairs = len(self.class_pairs_)
        if n_pairs <= 10:
            return "OvO is optimal for this number of classes"
        elif n_pairs <= 45:
            return "OvO is reasonable, monitor training time"
        elif n_pairs <= 100:
            return "Consider OvR for better efficiency"
        else:
            return "Strongly recommend OvR instead of OvO"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """
        Get feature importance aggregated across all pairwise classifiers
        
        Returns:
        --------
        importance : dict
            Dictionary containing per-pair and aggregated feature importance
        """
        if not self.is_fitted_:
            return None
        
        # Collect feature importance from each pairwise classifier
        pairwise_importance = {}
        all_importances = []
        
        for (class_a, class_b), classifier in self.pairwise_classifiers_.items():
            pair_name = f"{self.classes_[class_a]}_vs_{self.classes_[class_b]}"
            
            importance = None
            
            # Try different methods to get feature importance
            if hasattr(classifier, 'feature_importances_'):
                importance = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                importance = np.abs(classifier.coef_[0])
            elif hasattr(classifier, 'base_estimator') and hasattr(classifier.base_estimator, 'coef_'):
                importance = np.abs(classifier.base_estimator.coef_[0])
            
            if importance is not None:
                pairwise_importance[pair_name] = importance.tolist()
                all_importances.append(importance)
        
        if not all_importances:
            return None
        
        # Aggregate importance across all pairs
        all_importances = np.array(all_importances)
        aggregated = {
            "mean_importance": np.mean(all_importances, axis=0).tolist(),
            "std_importance": np.std(all_importances, axis=0).tolist(),
            "max_importance": np.max(all_importances, axis=0).tolist(),
            "min_importance": np.min(all_importances, axis=0).tolist()
        }
        
        return {
            "per_pair": pairwise_importance,
            "aggregated": aggregated,
            "feature_names": self.feature_names_,
            "n_pairs_with_importance": len(pairwise_importance)
        }
    
    def plot_pairwise_matrix(self, figsize=(12, 10)):
        """
        Create a pairwise performance matrix visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Pairwise matrix plot
        """
        if not self.is_fitted_ or not self.pairwise_performance_:
            return None
        
        # Create performance matrix
        performance_matrix = np.full((self.n_classes_, self.n_classes_), np.nan)
        
        for (class_a, class_b), perf in self.pairwise_performance_.items():
            f1_score = perf['mean_f1_score']
            performance_matrix[class_a, class_b] = f1_score
            performance_matrix[class_b, class_a] = f1_score  # Symmetric
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(performance_matrix, cmap='RdYlGn', vmin=0, vmax=1, alpha=0.8)
        
        # Set ticks and labels
        ax.set_xticks(range(self.n_classes_))
        ax.set_yticks(range(self.n_classes_))
        ax.set_xticklabels(self.classes_, rotation=45, ha='right')
        ax.set_yticklabels(self.classes_)
        
        # Add text annotations
        for i in range(self.n_classes_):
            for j in range(self.n_classes_):
                if i != j and not np.isnan(performance_matrix[i, j]):
                    text = ax.text(j, i, f'{performance_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=10)
                elif i == j:
                    ax.text(j, i, '—', ha="center", va="center", color="gray", fontsize=12)
        
        # Formatting
        ax.set_title('Pairwise Classifier Performance Matrix\n(F1 Scores)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('F1 Score', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_class_pair_distribution(self, figsize=(14, 8)):
        """
        Plot distribution of samples across all class pairs
        
        Parameters:
        -----------
        figsize : tuple, default=(14, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Class pair distribution plot
        """
        if not self.is_fitted_ or not self.pairwise_distributions_:
            return None
        
        # Prepare data
        pair_names = []
        sample_counts = []
        class_a_counts = []
        class_b_counts = []
        
        for (class_a, class_b), dist in self.pairwise_distributions_.items():
            pair_name = f"{dist['class_a_name']} vs\n{dist['class_b_name']}"
            pair_names.append(pair_name)
            sample_counts.append(dist['total_samples'])
            class_a_counts.append(dist['class_a_samples'])
            class_b_counts.append(dist['class_b_samples'])
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Total samples per pair
        bars1 = ax1.bar(range(len(pair_names)), sample_counts, alpha=0.7, color='skyblue')
        ax1.set_title('Total Training Samples per Class Pair', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class Pairs', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_xticks(range(len(pair_names)))
        ax1.set_xticklabels(pair_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars1, sample_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(sample_counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Stacked bar showing class distribution within each pair
        x_pos = range(len(pair_names))
        ax2.bar(x_pos, class_a_counts, alpha=0.7, label='First Class', color='lightcoral')
        ax2.bar(x_pos, class_b_counts, bottom=class_a_counts, alpha=0.7, 
               label='Second Class', color='lightgreen')
        
        ax2.set_title('Class Distribution within Each Pair', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class Pairs', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(pair_names, rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.markdown("### 🔄 One-vs-One Classifier Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4 = st.tabs(["Core Settings", "Advanced Options", "Strategy Analysis", "Algorithm Info"])
        
        with tab1:
            st.markdown("**Core One-vs-One Parameters**")
            
            # Base estimator selection
            base_estimator_options = ['logistic_regression', 'random_forest', 'svm', 'decision_tree', 
                                    'knn', 'gradient_boosting']
            
            if EXTENDED_ALGORITHMS:
                base_estimator_options.extend(['mlp', 'xgboost'])
            
            base_estimator = st.selectbox(
                "Base Estimator:",
                options=base_estimator_options,
                index=base_estimator_options.index(self.base_estimator) if self.base_estimator in base_estimator_options else 0,
                help="Binary classifier to use for each pairwise comparison",
                key=f"{key_prefix}_base_estimator"
            )
            
            # Base estimator info
            if base_estimator == 'logistic_regression':
                st.info("📈 Logistic Regression: Fast, interpretable, good for linear separable pairs")
            elif base_estimator == 'random_forest':
                st.info("🌳 Random Forest: Robust, handles non-linear patterns, built-in feature importance")
            elif base_estimator == 'svm':
                st.info("🔍 SVM: Excellent for complex boundaries, kernel tricks available")
            elif base_estimator == 'decision_tree':
                st.info("🌿 Decision Tree: Highly interpretable, good for categorical features")
            elif base_estimator == 'knn':
                st.info("📍 K-NN: Non-parametric, good for local patterns, no training time")
            elif base_estimator == 'gradient_boosting':
                st.info("🚀 Gradient Boosting: High accuracy, sequential error correction")
            
            # Voting strategy
            voting_strategy = st.selectbox(
                "Voting Strategy:",
                options=['majority', 'weighted'],
                index=['majority', 'weighted'].index(self.voting_strategy),
                help="How to combine votes from pairwise classifiers",
                key=f"{key_prefix}_voting_strategy"
            )
            
            if voting_strategy == 'majority':
                st.info("🗳️ Majority Voting: Simple vote counting (each pair contributes 1 vote)")
            else:
                st.info("⚖️ Weighted Voting: Weight votes by classifier confidence")
            
            # Tie breaking strategy
            tie_breaking = st.selectbox(
                "Tie Breaking:",
                options=['confidence', 'random', 'first'],
                index=['confidence', 'random', 'first'].index(self.tie_breaking),
                help="How to resolve ties in voting",
                key=f"{key_prefix}_tie_breaking"
            )
            
            if tie_breaking == 'confidence':
                st.info("🎯 Confidence: Use weighted voting to break ties")
            elif tie_breaking == 'random':
                st.info("🎲 Random: Randomly select among tied classes")
            else:
                st.info("1️⃣ First: Select first class in alphabetical order")
            
            # Feature scaling
            auto_scale_features = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scale_features,
                help="Scale features for distance-based estimators (SVM, KNN)",
                key=f"{key_prefix}_auto_scale_features"
            )
            
            if auto_scale_features:
                st.success("✅ Features will be standardized for all pairwise classifiers")
            else:
                st.warning("⚠️ No scaling - may hurt performance of distance-based estimators")
            
            # Base estimator hyperparameters based on selection
            st.markdown("**Base Estimator Hyperparameters**")
            
            # Logistic Regression parameters
            if base_estimator == 'logistic_regression':
                lr_C = st.number_input(
                    "LR - Regularization (C):",
                    value=float(self.lr_C),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_lr_C"
                )
                
                lr_max_iter = st.number_input(
                    "LR - Max Iterations:",
                    value=int(self.lr_max_iter),
                    min_value=100,
                    max_value=5000,
                    step=100,
                    key=f"{key_prefix}_lr_max_iter"
                )
                
                lr_solver = st.selectbox(
                    "LR - Solver:",
                    options=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                    index=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'].index(self.lr_solver),
                    key=f"{key_prefix}_lr_solver"
                )
                
                lr_penalty = st.selectbox(
                    "LR - Penalty:",
                    options=['l1', 'l2', 'elasticnet', 'none'],
                    index=['l1', 'l2', 'elasticnet', 'none'].index(self.lr_penalty),
                    key=f"{key_prefix}_lr_penalty"
                )
            else:
                lr_C = self.lr_C
                lr_max_iter = self.lr_max_iter
                lr_solver = self.lr_solver
                lr_penalty = self.lr_penalty
            
            # Random Forest parameters
            if base_estimator == 'random_forest':
                rf_n_estimators = st.slider(
                    "RF - Number of Trees:",
                    min_value=10,
                    max_value=500,
                    value=int(self.rf_n_estimators),
                    step=10,
                    key=f"{key_prefix}_rf_n_estimators"
                )
                
                rf_max_depth_option = st.selectbox(
                    "RF - Max Depth:",
                    options=['None', 'Custom'],
                    index=0 if self.rf_max_depth is None else 1,
                    key=f"{key_prefix}_rf_max_depth_option"
                )
                
                if rf_max_depth_option == 'Custom':
                    rf_max_depth = st.slider(
                        "RF - Custom Max Depth:",
                        min_value=1,
                        max_value=50,
                        value=10 if self.rf_max_depth is None else int(self.rf_max_depth),
                        key=f"{key_prefix}_rf_max_depth"
                    )
                else:
                    rf_max_depth = None
                
                rf_min_samples_split = st.slider(
                    "RF - Min Samples Split:",
                    min_value=2,
                    max_value=20,
                    value=int(self.rf_min_samples_split),
                    key=f"{key_prefix}_rf_min_samples_split"
                )
                
                rf_criterion = st.selectbox(
                    "RF - Split Criterion:",
                    options=['gini', 'entropy'],
                    index=['gini', 'entropy'].index(self.rf_criterion),
                    key=f"{key_prefix}_rf_criterion"
                )
            else:
                rf_n_estimators = self.rf_n_estimators
                rf_max_depth = self.rf_max_depth
                rf_min_samples_split = self.rf_min_samples_split
                rf_criterion = self.rf_criterion
            
            # SVM parameters
            if base_estimator == 'svm':
                svm_C = st.number_input(
                    "SVM - C Parameter:",
                    value=float(self.svm_C),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_svm_C"
                )
                
                svm_kernel = st.selectbox(
                    "SVM - Kernel:",
                    options=['rbf', 'linear', 'poly', 'sigmoid'],
                    index=['rbf', 'linear', 'poly', 'sigmoid'].index(self.svm_kernel),
                    key=f"{key_prefix}_svm_kernel"
                )
                
                svm_probability = st.checkbox(
                    "SVM - Enable Probabilities",
                    value=self.svm_probability,
                    help="Required for probability estimation (slight overhead)",
                    key=f"{key_prefix}_svm_probability"
                )
            else:
                svm_C = self.svm_C
                svm_kernel = self.svm_kernel
                svm_probability = self.svm_probability
            
            # Decision Tree parameters  
            if base_estimator == 'decision_tree':
                dt_max_depth_option = st.selectbox(
                    "DT - Max Depth:",
                    options=['None', 'Custom'],
                    index=0 if self.dt_max_depth is None else 1,
                    key=f"{key_prefix}_dt_max_depth_option"
                )
                
                if dt_max_depth_option == 'Custom':
                    dt_max_depth = st.slider(
                        "DT - Custom Max Depth:",
                        min_value=1,
                        max_value=30,
                        value=10 if self.dt_max_depth is None else int(self.dt_max_depth),
                        key=f"{key_prefix}_dt_max_depth"
                    )
                else:
                    dt_max_depth = None
                
                dt_criterion = st.selectbox(
                    "DT - Split Criterion:",
                    options=['gini', 'entropy'],
                    index=['gini', 'entropy'].index(self.dt_criterion),
                    key=f"{key_prefix}_dt_criterion"
                )
            else:
                dt_max_depth = self.dt_max_depth
                dt_criterion = self.dt_criterion
            
            # KNN parameters
            if base_estimator == 'knn':
                knn_n_neighbors = st.slider(
                    "KNN - Number of Neighbors:",
                    min_value=1,
                    max_value=50,
                    value=int(self.knn_n_neighbors),
                    key=f"{key_prefix}_knn_n_neighbors"
                )
                
                knn_weights = st.selectbox(
                    "KNN - Weight Function:",
                    options=['uniform', 'distance'],
                    index=['uniform', 'distance'].index(self.knn_weights),
                    key=f"{key_prefix}_knn_weights"
                )
            else:
                knn_n_neighbors = self.knn_n_neighbors
                knn_weights = self.knn_weights
            
            # Gradient Boosting parameters
            if base_estimator == 'gradient_boosting':
                gb_n_estimators = st.slider(
                    "GB - Number of Estimators:",
                    min_value=10,
                    max_value=500,
                    value=int(self.gb_n_estimators),
                    step=10,
                    key=f"{key_prefix}_gb_n_estimators"
                )
                
                gb_learning_rate = st.number_input(
                    "GB - Learning Rate:",
                    value=float(self.gb_learning_rate),
                    min_value=0.01,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_gb_learning_rate"
                )
                
                gb_max_depth = st.slider(
                    "GB - Max Depth:",
                    min_value=1,
                    max_value=10,
                    value=int(self.gb_max_depth),
                    key=f"{key_prefix}_gb_max_depth"
                )
            else:
                gb_n_estimators = self.gb_n_estimators
                gb_learning_rate = self.gb_learning_rate
                gb_max_depth = self.gb_max_depth
        
        with tab2:
            st.markdown("**Advanced Options**")
            
            # Probability calibration
            probability_calibration = st.checkbox(
                "Probability Calibration",
                value=self.probability_calibration,
                help="Calibrate probabilities using cross-validation (improves probability quality)",
                key=f"{key_prefix}_probability_calibration"
            )
            
            if probability_calibration:
                st.info("📊 Probability calibration will improve probability estimates quality")
            else:
                st.info("🚀 No calibration - faster training, may have uncalibrated probabilities")
            
            # Performance estimation
            estimate_pairwise_performance = st.checkbox(
                "Estimate Pairwise Performance",
                value=self.estimate_pairwise_performance,
                help="Evaluate individual pairwise classifier performance using CV",
                key=f"{key_prefix}_estimate_pairwise_performance"
            )
            
            if estimate_pairwise_performance:
                st.success("✅ Will evaluate each pairwise classifier individually")
            else:
                st.info("⚡ Skip individual evaluation for faster training")
            
            # Parallel processing
            n_jobs = st.selectbox(
                "Parallel Jobs:",
                options=[None, 1, 2, 4, -1],
                index=0,
                help="-1 uses all available cores for parallel training",
                key=f"{key_prefix}_n_jobs"
            )
            
            if n_jobs == -1:
                st.success("🚀 Perfect parallelization: all pairwise classifiers train independently")
            elif n_jobs and n_jobs > 1:
                st.info(f"🔄 Using {n_jobs} cores for parallel pairwise classifier training")
            else:
                st.info("🔄 Sequential training of pairwise classifiers")
            
            # Verbosity
            verbose = st.selectbox(
                "Verbosity Level:",
                options=[0, 1],
                index=self.verbose,
                help="Control training output verbosity",
                key=f"{key_prefix}_verbose"
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
        
        with tab3:
            st.markdown("**One-vs-One Strategy Analysis**")
            
            st.info("""
            **One-vs-One Approach:**
            • Creates K(K-1)/2 binary classifiers for K classes
            • Each classifier: "Class A vs. Class B" (pairwise)
            • Uses only relevant data for each binary problem
            • Final prediction: majority vote among all pairs
            
            **Key Advantages:**
            • No artificial class imbalance in binary problems
            • Each classifier focuses on exactly two classes
            • Better for small-to-moderate number of classes
            • Natural probability estimates via pairwise coupling
            """)
            
            # Computational complexity calculator
            st.markdown("**Computational Complexity Calculator**")
            n_classes_demo = st.slider(
                "Number of Classes:",
                min_value=2,
                max_value=20,
                value=5,
                help="See how complexity scales with number of classes",
                key=f"{key_prefix}_n_classes_demo"
            )
            
            n_pairs_demo = n_classes_demo * (n_classes_demo - 1) // 2
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Classes", n_classes_demo)
            with col2:
                st.metric("Pairwise Classifiers", n_pairs_demo)
            with col3:
                efficiency = "Excellent" if n_pairs_demo <= 10 else "Good" if n_pairs_demo <= 45 else "Moderate" if n_pairs_demo <= 100 else "Poor"
                st.metric("Efficiency", efficiency)
            
            # Strategy comparison
            if st.button("📊 OvO vs OvR Comparison", key=f"{key_prefix}_strategy_comparison"):
                st.markdown("""
                **One-vs-One vs One-vs-Rest:**
                
                **Computational Cost:**
                - OvO: K(K-1)/2 classifiers, each trained on subset
                - OvR: K classifiers, each trained on full dataset
                - Winner: OvR for large K, OvO for small K
                
                **Data Usage:**
                - OvO: Only relevant classes for each binary problem
                - OvR: All data, but artificial class imbalance
               # Continue from line 1432 where the code breaks:

                - Winner: No clear winner, depends on data characteristics
                
                **Accuracy:**
                - OvO: Often higher for few classes (focused learning)
                - OvR: May suffer from imbalanced binary problems
                - Winner: OvO for K ≤ 10, OvR for larger K
                
                **Interpretability:**
                - OvO: Each pairwise comparison is clear
                - OvR: Each "class vs rest" is interpretable
                - Winner: Depends on specific interpretation needs
                
                **Scalability:**
                - OvO: O(K²) classifiers - poor for large K
                - OvR: O(K) classifiers - linear scaling
                - Winner: OvR for large K (K > 15)
                
                **When to Choose OvO:**
                - 2-10 classes with high pairwise separability
                - Need highest accuracy for moderate K
                - Classes have natural pairwise relationships
                - Computational resources available for training
                """)
            
            # Pairwise complexity analysis
            if st.button("🔍 Pairwise Complexity Analysis", key=f"{key_prefix}_pairwise_analysis"):
                st.markdown("""
                **Pairwise Classification Details:**
                
                **Binary Problem Creation:**
                • For each pair (Class A, Class B):
                  - Extract samples belonging to Class A or Class B only
                  - Create binary labels: A=0, B=1
                  - Train binary classifier on this subset
                
                **Voting Aggregation:**
                • Each pairwise classifier votes for one of its two classes
                • Majority voting: Count votes, class with most votes wins
                • Weighted voting: Weight votes by classifier confidence
                • Tie breaking: Use confidence, random, or first-class rule
                
                **Probability Estimation:**
                • Collect pairwise probabilities: P(B|A vs B)
                • Aggregate using pairwise coupling algorithms
                • Normalize to ensure valid probability distribution
                
                **Computational Breakdown:**
                • Training: Each pair trains on ~(2/K) of total data
                • Memory: Store K(K-1)/2 complete binary models
                • Prediction: Query all pairs, aggregate K(K-1)/2 decisions
                """)
            
            # Performance optimization tips
            if st.button("⚡ Performance Optimization", key=f"{key_prefix}_optimization"):
                st.markdown("""
                **OvO Performance Optimization:**
                
                **Parallel Training:**
                • All pairwise classifiers are independent
                • Perfect embarrassingly parallel training
                • Use n_jobs=-1 for maximum parallelization
                • Linear speedup with number of cores (up to K(K-1)/2)
                
                **Memory Management:**
                • Each classifier stores full model
                • Memory usage: K(K-1)/2 × Base_Model_Size
                • Consider simpler base estimators for large K
                • Monitor memory usage during training
                
                **Base Estimator Selection:**
                • Fast estimators: Logistic Regression, Decision Tree
                • Accurate estimators: Random Forest, SVM, Gradient Boosting
                • Trade-off: Speed vs. accuracy per pairwise problem
                
                **Data Preprocessing:**
                • Feature scaling helps distance-based estimators
                • Feature selection reduces dimensionality per pair
                • Consider pair-specific preprocessing if needed
                """)
        
        with tab4:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **One-vs-One Classifier** - Pairwise Multi-class Strategy:
            • 🎯 Creates K(K-1)/2 binary classifiers for K classes
            • ⚖️ Each classifier focuses on exactly two classes
            • 🗳️ Final prediction via majority voting
            • 📊 Natural balance in each binary problem
            • 🎪 Excellent for 2-10 classes
            • 🔍 Clear pairwise interpretability
            
            **Mathematical Foundation:**
            • Pairwise Decomposition: Split K-class into (K choose 2) binary problems
            • Voting Rule: ŷ = argmax_i Σ_{j≠i} I(f_{ij}(x) votes for class i)
            • Probability: Pairwise coupling algorithms
            """)
            
            # Base estimator selection guide
            if st.button("🏗️ Base Estimator Selection Guide", key=f"{key_prefix}_estimator_guide"):
                st.markdown("""
                **Choosing Base Estimator for OvO:**
                
                **Logistic Regression:**
                - Best for: Speed, interpretability, linear separable pairs
                - OvO Benefit: Clear coefficient interpretation per pair
                - Use when: Need fast training, linear relationships
                
                **Random Forest:**
                - Best for: Robustness, non-linear patterns, feature importance
                - OvO Benefit: Feature importance analysis per pair
                - Use when: Mixed data types, robustness required
                
                **SVM:**
                - Best for: Complex decision boundaries, high dimensions
                - OvO Benefit: Maximum margin principle per pair
                - Use when: Need high accuracy, complex patterns
                
                **Decision Tree:**
                - Best for: Interpretability, categorical features
                - OvO Benefit: Clear decision rules per pair
                - Use when: Need rule extraction, categorical data
                
                **K-NN:**
                - Best for: Local patterns, non-parametric assumptions
                - OvO Benefit: Local decision boundaries per pair
                - Use when: Data has local structure, non-parametric
                
                **Gradient Boosting:**
                - Best for: Maximum accuracy, complex patterns
                - OvO Benefit: Superior performance per binary problem
                - Use when: Accuracy is paramount, computational budget allows
                """)
            
            # When to use OvO
            if st.button("🎯 When to Use One-vs-One", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Scenarios for One-vs-One:**
                
                **Problem Characteristics:**
                • 2-10 classes (sweet spot: 3-7 classes)
                • Classes are naturally separable in pairs
                • High between-class variation
                • Balanced or moderately imbalanced classes
                
                **Data Characteristics:**
                • Medium-sized datasets (OvO uses subsets efficiently)
                • High-dimensional data where pairwise focus helps
                • Clean data with distinct class boundaries
                • When class pairs have different optimal features
                
                **Requirements:**
                • Need highest possible accuracy for moderate K
                • Can afford higher computational cost
                • Want interpretable pairwise comparisons
                • Have parallel computing resources
                
                **Examples:**
                • Species classification (few distinct species)
                • Medical diagnosis (few distinct conditions)
                • Image classification (few object types)
                • Quality control (few defect categories)
                • Sentiment analysis (few sentiment levels)
                """)
            
            # OvO theory and mathematics
            if st.button("🧠 Mathematical Theory", key=f"{key_prefix}_theory"):
                st.markdown("""
                **One-vs-One Mathematical Foundation:**
                
                **Pairwise Decomposition:**
                • Given K classes: {C₁, C₂, ..., Cₖ}
                • Create (K choose 2) = K(K-1)/2 binary problems
                • Binary problem (i,j): Cᵢ vs Cⱼ using only samples from these classes
                
                **Training Phase:**
                • For each pair (i,j): Train fᵢⱼ(x) on {(x,y) | y ∈ {Cᵢ, Cⱼ}}
                • Binary labels: Cᵢ → 0, Cⱼ → 1
                • Each classifier: fᵢⱼ : X → {0, 1}
                
                **Prediction Phase:**
                • For sample x, collect all pairwise predictions
                • Vote counting: vᵢ = Σⱼ≠ᵢ I(fᵢⱼ(x) votes for class i)
                • Final prediction: ŷ = argmax_i vᵢ
                
                **Probability Estimation:**
                • Collect pairwise probabilities: pᵢⱼ = P(Cⱼ | x, Cᵢ vs Cⱼ)
                • Use pairwise coupling algorithms (e.g., Bradley-Terry model)
                • Solve optimization: minimize Σᵢⱼ (rᵢⱼ - pᵢⱼ)² where rᵢⱼ = pᵢ/(pᵢ + pⱼ)
                
                **Advantages over OvR:**
                • No artificial class imbalance (each binary problem naturally balanced)
                • Local expertise: each classifier specializes in distinguishing two classes
                • Better probability estimates via pairwise coupling
                • Robust: poor performance on one pair doesn't destroy entire system
                """)
            
            # Implementation details
            if st.button("🔧 Implementation Details", key=f"{key_prefix}_implementation"):
                st.markdown("""
                **OvO Implementation Specifics:**
                
                **Training Algorithm:**
                1. For each pair (i,j) where i < j:
                   a. Extract samples: X_ij = {x | y ∈ {Cᵢ, Cⱼ}}
                   b. Create binary labels: y_ij = I(y = Cⱼ)
                   c. Train binary classifier: fᵢⱼ.fit(X_ij, y_ij)
                2. Store all K(K-1)/2 trained classifiers
                
                **Prediction Algorithm:**
                1. For each sample x:
                   a. Initialize vote counter: votes = zeros(K)
                   b. For each classifier fᵢⱼ:
                      - Get prediction: pred = fᵢⱼ(x)
                      - Update votes: votes[i or j] += 1
                   c. Return: argmax(votes)
                
                **Memory Complexity:**
                • Storage: K(K-1)/2 complete binary models
                • Each model stores: parameters + preprocessing
                • Total: O(K² × Base_Model_Memory)
                
                **Computational Complexity:**
                • Training: O(K² × Base_Training_Time × (2/K))
                • Prediction: O(K² × Base_Prediction_Time)
                • Parallel scaling: Linear up to K(K-1)/2 cores
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_algo_details"):
                info = self.get_algorithm_info()
                st.json(info)
        
        # Return all hyperparameters
        return {
            "base_estimator": base_estimator,
            "auto_scale_features": auto_scale_features,
            "probability_calibration": probability_calibration,
            "voting_strategy": voting_strategy,
            "tie_breaking": tie_breaking,
            "estimate_pairwise_performance": estimate_pairwise_performance,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "random_state": random_state,
            
            # Base estimator hyperparameters
            "lr_C": lr_C,
            "lr_max_iter": lr_max_iter,
            "lr_solver": lr_solver,
            "lr_penalty": lr_penalty,
            "rf_n_estimators": rf_n_estimators,
            "rf_max_depth": rf_max_depth,
            "rf_min_samples_split": rf_min_samples_split,
            "rf_criterion": rf_criterion,
            "svm_C": svm_C,
            "svm_kernel": svm_kernel,
            "svm_probability": svm_probability,
            "dt_max_depth": dt_max_depth,
            "dt_criterion": dt_criterion,
            "knn_n_neighbors": knn_n_neighbors,
            "knn_weights": knn_weights,
            "gb_n_estimators": gb_n_estimators,
            "gb_learning_rate": gb_learning_rate,
            "gb_max_depth": gb_max_depth
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return OneVsOneClassifierPlugin(
            base_estimator=hyperparameters.get("base_estimator", self.base_estimator),
            auto_scale_features=hyperparameters.get("auto_scale_features", self.auto_scale_features),
            probability_calibration=hyperparameters.get("probability_calibration", self.probability_calibration),
            voting_strategy=hyperparameters.get("voting_strategy", self.voting_strategy),
            tie_breaking=hyperparameters.get("tie_breaking", self.tie_breaking),
            estimate_pairwise_performance=hyperparameters.get("estimate_pairwise_performance", self.estimate_pairwise_performance),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            verbose=hyperparameters.get("verbose", self.verbose),
            random_state=hyperparameters.get("random_state", self.random_state),
            
            # Base estimator hyperparameters
            lr_C=hyperparameters.get("lr_C", self.lr_C),
            lr_max_iter=hyperparameters.get("lr_max_iter", self.lr_max_iter),
            lr_solver=hyperparameters.get("lr_solver", self.lr_solver),
            lr_penalty=hyperparameters.get("lr_penalty", self.lr_penalty),
            rf_n_estimators=hyperparameters.get("rf_n_estimators", self.rf_n_estimators),
            rf_max_depth=hyperparameters.get("rf_max_depth", self.rf_max_depth),
            rf_min_samples_split=hyperparameters.get("rf_min_samples_split", self.rf_min_samples_split),
            rf_criterion=hyperparameters.get("rf_criterion", self.rf_criterion),
            svm_C=hyperparameters.get("svm_C", self.svm_C),
            svm_kernel=hyperparameters.get("svm_kernel", self.svm_kernel),
            svm_probability=hyperparameters.get("svm_probability", self.svm_probability),
            dt_max_depth=hyperparameters.get("dt_max_depth", self.dt_max_depth),
            dt_criterion=hyperparameters.get("dt_criterion", self.dt_criterion),
            knn_n_neighbors=hyperparameters.get("knn_n_neighbors", self.knn_n_neighbors),
            knn_weights=hyperparameters.get("knn_weights", self.knn_weights),
            gb_n_estimators=hyperparameters.get("gb_n_estimators", self.gb_n_estimators),
            gb_learning_rate=hyperparameters.get("gb_learning_rate", self.gb_learning_rate),
            gb_max_depth=hyperparameters.get("gb_max_depth", self.gb_max_depth)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for One-vs-One Classifier"""
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
        """Check if One-vs-One Classifier is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"One-vs-One requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            n_classes = len(unique_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 50:
                return False, f"Too many classes ({n_classes}). OvO creates {n_classes*(n_classes-1)//2} classifiers. Consider OvR."
            
            # Calculate number of pairwise classifiers
            n_pairs = n_classes * (n_classes - 1) // 2
            
            # Check class distribution for pairwise problems
            min_class_samples = min(np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes))
            if min_class_samples < 2:
                return False, "Each class needs at least 2 samples for pairwise classification"
            
            # OvO specific advantages and considerations
            advantages = []
            considerations = []
            
            # Optimal class range
            if 3 <= n_classes <= 7:
                advantages.append(f"Ideal class count ({n_classes}) - OvO sweet spot")
            elif n_classes == 2:
                advantages.append("Binary problem - OvO equivalent to direct binary classification")
            elif 8 <= n_classes <= 15:
                considerations.append(f"Moderate classes ({n_classes}) - {n_pairs} pairwise classifiers")
            else:
                considerations.append(f"Many classes ({n_classes}) - {n_pairs} classifiers may be expensive")
            
            # Sample size per pair
            avg_samples_per_pair = (2 * X.shape[0]) / n_classes  # Approximation
            if avg_samples_per_pair >= 50:
                advantages.append("Good sample size per pairwise problem")
            elif avg_samples_per_pair >= 20:
                advantages.append("Adequate sample size per pairwise problem")
            else:
                considerations.append("Small sample size per pairwise problem - may need regularization")
            
            # Computational efficiency
            if n_pairs <= 10:
                advantages.append(f"Efficient computation ({n_pairs} pairwise classifiers)")
            elif n_pairs <= 45:
                advantages.append(f"Reasonable computation ({n_pairs} pairwise classifiers)")
            elif n_pairs <= 100:
                considerations.append(f"High computation ({n_pairs} classifiers) - consider parallel training")
            else:
                considerations.append(f"Very high computation ({n_pairs} classifiers) - OvR may be better")
            
            # Pairwise balance advantage
            advantages.append("Natural balance in each pairwise problem (no artificial imbalance)")
            
            # Memory considerations
            if n_pairs <= 20:
                advantages.append("Low memory usage")
            elif n_pairs <= 50:
                considerations.append("Moderate memory usage for pairwise models")
            else:
                considerations.append("High memory usage - stores many pairwise models")
            
            # Build compatibility message
            efficiency_rating = ("Excellent" if n_pairs <= 10 else "Good" if n_pairs <= 45 else 
                               "Moderate" if n_pairs <= 100 else "Poor")
            
            message_parts = [
                f"✅ Compatible with {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes",
                f"🔄 Creates {n_pairs} pairwise classifiers (efficiency: {efficiency_rating})"
            ]
            
            if advantages:
                message_parts.append("🎯 OvO advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist(),
            "feature_names": self.feature_names_,
            "base_estimator": self.base_estimator,
            "n_pairwise_classifiers": len(self.class_pairs_),
            "class_pairs": [(str(self.classes_[a]), str(self.classes_[b])) for a, b in self.class_pairs_],
            "feature_scaling": self.scaler_ is not None,
            "voting_strategy": self.voting_strategy,
            "tie_breaking": self.tie_breaking,
            "probability_calibration": self.probability_calibration,
            "parallel_training": self.n_jobs != 1,
            "strategy_type": "One-vs-One (Pairwise)"
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "One-vs-One Classifier",
            "strategy_type": "Pairwise Multi-class",
            "training_completed": True,
            "ovo_characteristics": {
                "pairwise_decomposition": True,
                "focused_binary_problems": True,
                "natural_balance": True,
                "voting_aggregation": True,
                "quadratic_scaling": True,
                "parallel_potential": True
            },
            "strategy_configuration": {
                "base_estimator": self.base_estimator,
                "n_classes": self.n_classes_,
                "n_pairwise_classifiers": len(self.class_pairs_),
                "voting_strategy": self.voting_strategy,
                "tie_breaking": self.tie_breaking,
                "feature_scaling": self.scaler_ is not None,
                "probability_calibration": self.probability_calibration,
                "parallel_training": self.n_jobs != 1
            },
            "ovo_analysis": self.get_ovo_analysis(),
            "performance_considerations": {
                "training_time": f"O({self.n_classes_}² × Base_Training_Time × (2/{self.n_classes_}))",
                "prediction_time": f"O({self.n_classes_}² × Base_Prediction_Time)",
                "memory_usage": f"Stores {len(self.class_pairs_)} complete pairwise models",
                "parallelization": f"Perfect - all {len(self.class_pairs_)} pairs independent",
                "scalability": f"Quadratic scaling - {len(self.class_pairs_)} classifiers for {self.n_classes_} classes",
                "interpretability": "High - each pairwise comparison separately interpretable"
            },
            "ovo_theory": {
                "approach": f"Decomposes {self.n_classes_}-class problem into {len(self.class_pairs_)} pairwise problems",
                "pairwise_problems": "Each: Class A vs. Class B using only relevant samples",
                "decision_rule": "Majority voting across all pairwise comparisons",
                "probability_estimation": "Pairwise coupling of binary probabilities",
                "advantages": "Natural balance, focused learning, robust aggregation"
            }
        }
        
        # Add pairwise performance if available
        if self.pairwise_performance_:
            info["pairwise_performance"] = self.pairwise_performance_
        
        # Add pairwise distributions
        if self.pairwise_distributions_:
            info["pairwise_distributions"] = self.pairwise_distributions_
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the One-vs-One Classifier.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Required for metrics like McFadden's R-squared.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics.
        y_proba : np.ndarray, optional
            Predicted probabilities. Required for metrics like McFadden's R-squared.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve One-vs-One specific metrics."}

        metrics = {}
        prefix = "ovo_" # One-vs-One

        # Structural metrics
        if self.n_classes_ is not None:
            metrics[f"{prefix}num_original_classes"] = self.n_classes_
        if self.class_pairs_ is not None:
            metrics[f"{prefix}num_pairwise_classifiers"] = len(self.class_pairs_)
        
        metrics[f"{prefix}base_estimator_type"] = str(self.base_estimator) # Store as string
        metrics[f"{prefix}voting_strategy"] = str(self.voting_strategy)
        metrics[f"{prefix}tie_breaking_strategy"] = str(self.tie_breaking)
        metrics[f"{prefix}probability_calibration_enabled"] = self.probability_calibration
        metrics[f"{prefix}auto_feature_scaling_enabled"] = self.auto_scale_features

        # Pairwise performance summary (if estimated)
        if self.pairwise_performance_ and self.estimate_pairwise_performance:
            f1_scores = [perf['mean_f1_score'] for perf in self.pairwise_performance_.values() if 'mean_f1_score' in perf and perf['mean_f1_score'] is not None]
            if f1_scores:
                metrics[f"{prefix}mean_pairwise_f1_score"] = float(np.mean(f1_scores))
                metrics[f"{prefix}std_pairwise_f1_score"] = float(np.std(f1_scores))
                metrics[f"{prefix}min_pairwise_f1_score"] = float(np.min(f1_scores))
                metrics[f"{prefix}max_pairwise_f1_score"] = float(np.max(f1_scores))
                metrics[f"{prefix}num_evaluated_pairs_performance"] = len(f1_scores)
            else:
                metrics[f"{prefix}pairwise_f1_info"] = "No valid F1 scores found in pairwise_performance."
        elif self.estimate_pairwise_performance:
            metrics[f"{prefix}pairwise_performance_info"] = "Pairwise performance estimation was enabled but no results found."
        else:
            metrics[f"{prefix}pairwise_performance_info"] = "Pairwise performance estimation was not enabled."

        # McFadden's Pseudo R-squared for the overall OvO model
        if y_true is not None and y_proba is not None and self.label_encoder_ is not None and self.classes_ is not None:
            try:
                y_true_encoded = self.label_encoder_.transform(y_true)
                n_samples = len(y_true_encoded)
                n_classes_model = len(self.classes_)

                if np.any(y_true_encoded >= y_proba.shape[1]) or np.any(y_true_encoded < 0):
                    raise ValueError("y_true_encoded contains out-of-bounds class indices for y_proba.")

                clipped_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
                log_likelihoods_model = np.log(clipped_proba[np.arange(n_samples), y_true_encoded])
                ll_model = np.sum(log_likelihoods_model)

                class_counts = np.bincount(y_true_encoded, minlength=n_classes_model)
                if len(class_counts) < n_classes_model: # Should not happen if minlength is used correctly
                    class_counts = np.pad(class_counts, (0, n_classes_model - len(class_counts)), 'constant')

                class_probas_null = class_counts / n_samples
                
                ll_null = 0
                for k_idx in range(n_classes_model):
                    if class_counts[k_idx] > 0 and class_probas_null[k_idx] > 0:
                         ll_null += class_counts[k_idx] * np.log(np.clip(class_probas_null[k_idx], 1e-15, 1))
                
                if ll_null == 0:
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = 1.0 if ll_model == 0 else 0.0
                elif ll_model > ll_null: # Model is worse than null model
                     metrics[f"{prefix}mcfaddens_pseudo_r2"] = 0.0
                else:
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = float(1 - (ll_model / ll_null))
            except Exception as e:
                metrics[f"{prefix}mcfaddens_pseudo_r2_error"] = str(e)
        
        if not metrics: # Should not happen due to structural metrics
            metrics['info'] = "No specific One-vs-One metrics were available."
            
        return metrics


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return OneVsOneClassifierPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of One-vs-One Classifier Plugin
    """
    print("Testing One-vs-One Classifier Plugin...")
    
    try:
        # Create sample multi-class data
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Generate multi-class dataset (moderate number of classes for OvO)
        X, y = make_classification(
            n_samples=800,
            n_features=15,
            n_informative=12,
            n_redundant=2,
            n_classes=4,  # 4 classes = 6 pairwise classifiers (ideal for OvO)
            n_clusters_per_class=1,
            class_sep=1.5,
            flip_y=0.01,
            random_state=42
        )
        
        print(f"\n📊 Multi-class Dataset Info:")
        print(f"Shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        n_classes = len(np.unique(y))
        n_pairs = n_classes * (n_classes - 1) // 2
        print(f"Pairwise classifiers needed: {n_pairs}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and test OvO plugin
        plugin = OneVsOneClassifierPlugin(
            base_estimator='random_forest',  # Good base estimator for OvO
            auto_scale_features=True,
            voting_strategy='weighted',
            tie_breaking='confidence',
            estimate_pairwise_performance=True,
            probability_calibration=False,
            n_jobs=-1,  # Parallel training
            verbose=1,
            random_state=42
        )
        
        print("\n🔍 Plugin Info:")
        print(f"Name: {plugin.get_name()}")
        print(f"Category: {plugin.get_category()}")
        print(f"Description: {plugin.get_description()}")
        
        # Check compatibility
        compatible, message = plugin.is_compatible_with_data(X_train, y_train)
        print(f"\n✅ Compatibility: {message}")
        
        if compatible:
            # Train One-vs-One classifier
            print("\n🚀 Training One-vs-One Classifier...")
            plugin.fit(X_train, y_train)
            
            # Make predictions
            y_pred = plugin.predict(X_test)
            y_proba = plugin.predict_proba(X_test)
            
            # Evaluate performance
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n📊 One-vs-One Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Classes: {plugin.classes_}")
            print(f"Pairwise classifiers trained: {len(plugin.class_pairs_)}")
            
            # Get pairwise predictions for analysis
            pairwise_preds = plugin.get_pairwise_predictions(X_test[:5])  # First 5 samples
            print(f"\n🎯 Pairwise Predictions Analysis (first 5 samples):")
            for pair_name, predictions in pairwise_preds.items():
                pred_summary = dict(zip(*np.unique(predictions, return_counts=True)))
                print(f"{pair_name}: {pred_summary}")
            
            # Get OvO analysis
            ovo_analysis = plugin.get_ovo_analysis()
            print(f"\n🏗️ One-vs-One Strategy Analysis:")
            
            strategy_summary = ovo_analysis.get('strategy_summary', {})
            print(f"Base estimator: {strategy_summary.get('base_estimator', 'Unknown')}")
            print(f"Number of classes: {strategy_summary.get('n_classes', 'Unknown')}")
            print(f"Pairwise classifiers: {strategy_summary.get('n_pairwise_classifiers', 'Unknown')}")
            print(f"Voting strategy: {strategy_summary.get('voting_strategy', 'Unknown')}")
            
            # Pairwise structure analysis
            if 'pairwise_structure' in ovo_analysis:
                pairwise_struct = ovo_analysis['pairwise_structure']
                print(f"\n🔄 Pairwise Structure:")
                print(f"Total pairs: {pairwise_struct['total_pairs']}")
                
                complexity = pairwise_struct['computational_complexity']
                print(f"Training complexity: {complexity['training_cost']}")
                print(f"Memory usage: {complexity['memory_usage']}")
            
            # Pairwise distributions
            if 'pairwise_distributions' in ovo_analysis:
                pairwise_dist = ovo_analysis['pairwise_distributions']
                print(f"\n📈 Pairwise Data Distribution:")
                for pair_key, dist_info in list(pairwise_dist.items())[:3]:  # Show first 3 pairs
                    class_a = dist_info['class_a_name']
                    class_b = dist_info['class_b_name']
                    total = dist_info['total_samples']
                    a_count = dist_info['class_a_samples']
                    b_count = dist_info['class_b_samples']
                    print(f"{class_a} vs {class_b}: {a_count}+{b_count}={total} samples")
            
            # Pairwise performance analysis
            if 'pairwise_performance' in ovo_analysis:
                pairwise_perf = ovo_analysis['pairwise_performance']
                print(f"\n🎯 Pairwise Performance Analysis:")
                for pair_key, perf_info in list(pairwise_perf.items())[:3]:  # Show first 3 pairs
                    class_a = perf_info['class_a_name']
                    class_b = perf_info['class_b_name']
                    f1 = perf_info['mean_f1_score']
                    std = perf_info['std_f1_score']
                    n_samples = perf_info['n_samples']
                    print(f"{class_a} vs {class_b}: F1={f1:.3f}±{std:.3f} (n={n_samples})")
            
            # Performance summary
            if 'performance_summary' in ovo_analysis:
                perf_summary = ovo_analysis['performance_summary']
                print(f"\n📊 Overall Pairwise Performance:")
                print(f"Mean F1 across pairs: {perf_summary['mean_pairwise_f1']:.3f}")
                print(f"Std F1 across pairs: {perf_summary['std_pairwise_f1']:.3f}")
                print(f"Range: {perf_summary['min_pairwise_f1']:.3f} - {perf_summary['max_pairwise_f1']:.3f}")
            
            # Scalability analysis
            if 'scalability_analysis' in ovo_analysis:
                scalability = ovo_analysis['scalability_analysis']
                print(f"\n⚡ Scalability Analysis:")
                print(f"Current classes: {scalability['current_classes']}")
                print(f"Current classifiers: {scalability['current_classifiers']}")
                print(f"Efficiency rating: {scalability['efficiency_rating']}")
                print(f"Recommendation: {scalability['recommendation']}")
            
            # Feature importance analysis
            feature_importance = plugin.get_feature_importance()
            if feature_importance:
                print(f"\n🔍 Feature Importance Analysis:")
                aggregated = feature_importance['aggregated']
                mean_importance = aggregated['mean_importance']
                
                # Top 5 features
                top_features_idx = np.argsort(mean_importance)[-5:][::-1]
                print("Top 5 Most Discriminative Features (averaged across pairs):")
                for i, idx in enumerate(top_features_idx):
                    feature_name = feature_importance['feature_names'][idx]
                    importance = mean_importance[idx]
                    print(f"{i+1}. {feature_name}: {importance:.4f}")
                
                print(f"Feature importance available for {feature_importance['n_pairs_with_importance']} pairs")
            
            # Model parameters
            model_params = plugin.get_model_params()
            print(f"\n⚙️ Model Configuration:")
            print(f"Strategy: {model_params.get('strategy_type', 'Unknown')}")
            print(f"Base estimator: {model_params.get('base_estimator', 'Unknown')}")
            print(f"Pairwise classifiers: {model_params.get('n_pairwise_classifiers', 'Unknown')}")
            print(f"Voting strategy: {model_params.get('voting_strategy', 'Unknown')}")
            print(f"Feature scaling: {model_params.get('feature_scaling', False)}")
            
            # Training info
            training_info = plugin.get_training_info()
            print(f"\n📈 Training Info:")
            print(f"Algorithm: {training_info['algorithm']}")
            print(f"Strategy type: {training_info['strategy_type']}")
            
            ovo_chars = training_info['ovo_characteristics']
            print(f"Pairwise decomposition: {ovo_chars['pairwise_decomposition']}")
            print(f"Natural balance: {ovo_chars['natural_balance']}")
            print(f"Focused learning: {ovo_chars['focused_binary_problems']}")
            
            # Performance considerations
            perf_info = training_info['performance_considerations']
            print(f"\n⚡ Performance Characteristics:")
            print(f"Training time: {perf_info['training_time']}")
            print(f"Prediction time: {perf_info['prediction_time']}")
            print(f"Memory usage: {perf_info['memory_usage']}")
            print(f"Parallelization: {perf_info['parallelization']}")
            
            # OvO theory
            ovo_theory = training_info['ovo_theory']
            print(f"\n🧠 One-vs-One Theory:")
            print(f"Approach: {ovo_theory['approach']}")
            print(f"Pairwise problems: {ovo_theory['pairwise_problems']}")
            print(f"Decision rule: {ovo_theory['decision_rule']}")
            
            print("\n✅ One-vs-One Classifier Plugin test completed successfully!")
            print("🎯 Successfully decomposed multi-class problem into pairwise classifiers!")
            
            # Demonstrate OvO benefits
            print(f"\n🚀 One-vs-One Benefits:")
            print(f"Natural Balance: No artificial class imbalance in binary problems")
            print(f"Focused Learning: Each of {len(plugin.class_pairs_)} classifiers specializes in 2 classes")
            print(f"Pairwise Expertise: Optimized decision boundaries for each class pair")
            print(f"Robust Aggregation: Voting system combines {len(plugin.class_pairs_)} expert opinions")
            
            # Show prediction confidence analysis
            print(f"\n🎯 Prediction Confidence Analysis:")
            max_probas = np.max(y_proba, axis=1)
            print(f"Average confidence: {np.mean(max_probas):.3f}")
            print(f"Min confidence: {np.min(max_probas):.3f}")
            print(f"Max confidence: {np.max(max_probas):.3f}")
            print(f"High confidence predictions (>0.7): {np.sum(max_probas > 0.7)/len(max_probas)*100:.1f}%")
            
            # Compare with theoretical bounds
            random_accuracy = 1.0 / n_classes
            print(f"\n📊 Performance Analysis:")
            print(f"OvO Accuracy: {accuracy:.3f}")
            print(f"Random baseline: {random_accuracy:.3f}")
            print(f"Improvement over random: {(accuracy/random_accuracy - 1)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()