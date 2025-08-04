import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter

# Try to import optional libraries
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    EXTENDED_ALGORITHMS = True
except ImportError:
    EXTENDED_ALGORITHMS = False

# Import for plugin system
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin

class VotingClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Voting Classifier Plugin - Advanced Ensemble Method
    
    The Voting Classifier is a meta-estimator that combines multiple different algorithms
    and uses either hard voting (majority vote) or soft voting (average probabilities)
    to make final predictions. It's based on the principle that combining diverse models
    often leads to better performance than any individual model.
    
    Key Features:
    1. Hard Voting: Uses majority vote from base classifiers
    2. Soft Voting: Averages predicted probabilities (requires probability support)
    3. Flexible Base Estimator Selection: Choose from multiple algorithm types
    4. Automatic Weight Optimization: Can assign different weights to base estimators
    5. Cross-Validation Based Performance: Evaluates individual and ensemble performance
    """
    
    def __init__(self,
                 # Core voting parameters
                 voting='soft',
                 weights=None,
                 n_jobs=None,
                 flatten_transform=True,
                 verbose=False,
                 # Base estimator selection
                 use_logistic_regression=True,
                 use_random_forest=True,
                 use_svm=True,
                 use_knn=True,
                 use_naive_bayes=True,
                 use_decision_tree=False,
                 use_mlp=False,
                 use_gradient_boosting=False,
                 use_ada_boost=False,
                 use_lda=False,
                 use_qda=False,
                 # Hyperparameters for base estimators
                 lr_C=1.0,
                 lr_max_iter=1000,
                 rf_n_estimators=100,
                 rf_max_depth=None,
                 rf_random_state=42,
                 svm_C=1.0,
                 svm_kernel='rbf',
                 svm_probability=True,
                 knn_n_neighbors=5,
                 knn_weights='uniform',
                 dt_max_depth=None,
                 dt_random_state=42,
                 mlp_hidden_layer_sizes=(100,),
                 mlp_max_iter=500,
                 gb_n_estimators=100,
                 gb_learning_rate=0.1,
                 ada_n_estimators=50,
                 ada_learning_rate=1.0,
                 # Advanced options
                 auto_scale_features=True,
                 optimize_weights=False,
                 cross_validation_folds=5,
                 weight_optimization_method='accuracy',
                 random_state=42):
        """
        Initialize Voting Classifier with comprehensive configuration
        
        Parameters:
        -----------
        voting : str, default='soft'
            Voting strategy ('hard' or 'soft')
        weights : array-like, default=None
            Weights for base estimators
        n_jobs : int, default=None
            Number of jobs for parallel processing
        flatten_transform : bool, default=True
            Whether to flatten transform output
        verbose : bool, default=False
            Enable verbose output
        use_* : bool
            Flags to enable/disable specific base estimators
        *_* : various
            Hyperparameters for individual base estimators
        auto_scale_features : bool, default=True
            Whether to automatically scale features
        optimize_weights : bool, default=False
            Whether to optimize estimator weights using cross-validation
        cross_validation_folds : int, default=5
            Number of CV folds for weight optimization
        weight_optimization_method : str, default='accuracy'
            Metric for weight optimization ('accuracy', 'f1', 'roc_auc')
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__()
        
        # Core voting parameters
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose
        
        # Base estimator selection
        self.use_logistic_regression = use_logistic_regression
        self.use_random_forest = use_random_forest
        self.use_svm = use_svm
        self.use_knn = use_knn
        self.use_naive_bayes = use_naive_bayes
        self.use_decision_tree = use_decision_tree
        self.use_mlp = use_mlp
        self.use_gradient_boosting = use_gradient_boosting
        self.use_ada_boost = use_ada_boost
        self.use_lda = use_lda
        self.use_qda = use_qda
        
        # Hyperparameters for base estimators
        self.lr_C = lr_C
        self.lr_max_iter = lr_max_iter
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_random_state = rf_random_state
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_probability = svm_probability
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_weights = knn_weights
        self.dt_max_depth = dt_max_depth
        self.dt_random_state = dt_random_state
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_max_iter = mlp_max_iter
        self.gb_n_estimators = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate
        
        # Advanced options
        self.auto_scale_features = auto_scale_features
        self.optimize_weights = optimize_weights
        self.cross_validation_folds = cross_validation_folds
        self.weight_optimization_method = weight_optimization_method
        self.random_state = random_state
        
        # Plugin metadata
        self._name = "Voting Classifier"
        self._description = "Ensemble method that combines multiple algorithms using majority voting or probability averaging for robust predictions."
        self._category = "Ensemble Methods"
        self._algorithm_type = "Voting Ensemble"
        self._paper_reference = "Dietterich, T. G. (2000). Ensemble methods in machine learning. Multiple classifier systems, 1-15."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 50
        self._handles_missing_values = False
        self._requires_scaling = False  # Depends on base estimators
        self._supports_sparse = False  # Depends on base estimators
        self._is_linear = False  # Ensemble of various algorithms
        self._provides_feature_importance = True  # Can aggregate from base estimators
        self._provides_probabilities = True
        self._handles_categorical = False  # Depends on base estimators
        self._ensemble_method = True
        self._meta_algorithm = True
        self._combines_diverse_models = True
        self._voting_based = True
        self._probability_averaging = True
        
        # Internal attributes
        self.voting_classifier_ = None
        self.base_estimators_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.base_estimator_scores_ = None
        self.ensemble_score_ = None
        self.optimized_weights_ = None
        self.estimator_analysis_ = None
    
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
            "year_introduced": 1979,  # Early ensemble methods
            "key_innovations": {
                "ensemble_learning": "Combines multiple diverse algorithms",
                "voting_mechanisms": "Hard voting (majority) and soft voting (probability averaging)",
                "diversity_exploitation": "Leverages different algorithm strengths",
                "bias_variance_tradeoff": "Reduces variance through model averaging",
                "robust_predictions": "More stable than individual models",
                "meta_learning": "Learning how to combine base learners"
            },
            "algorithm_mechanics": {
                "training_process": {
                    "step_1": "Train each base estimator independently on training data",
                    "step_2": "Store trained models as ensemble components",
                    "step_3": "Optionally optimize weights using cross-validation",
                    "step_4": "Create final voting ensemble"
                },
                "prediction_process": {
                    "hard_voting": {
                        "step_1": "Get class predictions from each base estimator",
                        "step_2": "Apply optional weights to votes",
                        "step_3": "Take majority vote for final prediction",
                        "formula": "argmax_c Σ w_i * I(h_i(x) = c)"
                    },
                    "soft_voting": {
                        "step_1": "Get class probabilities from each base estimator",
                        "step_2": "Apply optional weights to probabilities",
                        "step_3": "Average weighted probabilities",
                        "step_4": "Take argmax for final prediction",
                        "formula": "argmax_c Σ w_i * P_i(c|x)"
                    }
                },
                "ensemble_theory": {
                    "condorcet_jury_theorem": "Majority voting improves accuracy if base classifiers are better than random",
                    "bias_variance_decomposition": "Ensemble reduces variance while maintaining similar bias",
                    "diversity_importance": "More diverse base learners lead to better ensemble performance",
                    "weight_optimization": "Optimal weights can be learned from validation data"
                }
            },
            "voting_strategies": {
                "hard_voting": {
                    "description": "Majority vote from class predictions",
                    "advantages": ["Simple and interpretable", "Works with any classifier", "Fast prediction"],
                    "disadvantages": ["Ignores prediction confidence", "Equal treatment of all models"],
                    "best_for": "When base estimators don't provide reliable probabilities",
                    "formula": "Final prediction = mode(predictions)"
                },
                "soft_voting": {
                    "description": "Weighted average of class probabilities",
                    "advantages": ["Uses prediction confidence", "Generally better performance", "Smooth decision boundaries"],
                    "disadvantages": ["Requires probability estimates", "More computationally expensive"],
                    "best_for": "When base estimators provide calibrated probabilities",
                    "formula": "Final prediction = argmax(average(probabilities))"
                }
            },
            "base_estimator_selection": {
                "diversity_principle": "Choose algorithms with different biases and assumptions",
                "recommended_combinations": [
                    "Linear (Logistic Regression) + Non-linear (Random Forest) + Instance-based (KNN)",
                    "Parametric (Naive Bayes) + Non-parametric (Decision Trees) + Kernel (SVM)",
                    "Fast (Naive Bayes) + Robust (Random Forest) + High-capacity (Neural Network)"
                ],
                "selection_criteria": {
                    "algorithmic_diversity": "Different learning paradigms",
                    "complementary_strengths": "Models good at different aspects",
                    "performance_threshold": "Each model should be reasonably good",
                    "computational_balance": "Mix of fast and complex models"
                }
            },
            "available_base_estimators": {
                "logistic_regression": {
                    "type": "Linear probabilistic",
                    "strengths": ["Fast", "Interpretable", "Good probabilities"],
                    "weaknesses": ["Linear assumptions", "Feature scaling sensitive"],
                    "best_for": "Linear patterns, high-dimensional data"
                },
                "random_forest": {
                    "type": "Tree ensemble",
                    "strengths": ["Handles non-linearity", "Feature importance", "Robust"],
                    "weaknesses": ["Can overfit", "Biased to categorical features"],
                    "best_for": "Complex patterns, mixed data types"
                },
                "svm": {
                    "type": "Kernel-based",
                    "strengths": ["Kernel trick", "Good generalization", "Works in high dimensions"],
                    "weaknesses": ["Slow on large data", "Parameter sensitive"],
                    "best_for": "High-dimensional data, complex decision boundaries"
                },
                "knn": {
                    "type": "Instance-based",
                    "strengths": ["Simple", "Non-parametric", "Local patterns"],
                    "weaknesses": ["Curse of dimensionality", "Sensitive to noise"],
                    "best_for": "Local patterns, irregular decision boundaries"
                },
                "naive_bayes": {
                    "type": "Probabilistic",
                    "strengths": ["Fast", "Small data", "Good probabilities"],
                    "weaknesses": ["Strong independence assumption"],
                    "best_for": "Text classification, small datasets"
                },
                "neural_network": {
                    "type": "Deep learning",
                    "strengths": ["Universal approximator", "Complex patterns"],
                    "weaknesses": ["Requires large data", "Black box"],
                    "best_for": "Complex non-linear patterns, large datasets"
                }
            },
            "strengths": [
                "Often outperforms individual base estimators",
                "Reduces overfitting through model averaging",
                "More robust and stable predictions",
                "Can combine different algorithm types",
                "Flexibility in choosing base estimators",
                "Both hard and soft voting options",
                "Can handle different types of features well",
                "Interpretable ensemble decisions",
                "Proven theoretical foundation",
                "Easy to implement and understand",
                "Can leverage existing model diversity",
                "Scales well with number of base estimators"
            ],
            "weaknesses": [
                "Increased computational cost (training multiple models)",
                "Higher memory requirements",
                "Slower prediction time",
                "May not improve if base estimators are similar",
                "Requires careful base estimator selection",
                "Soft voting requires probability calibration",
                "Can mask individual model interpretability",
                "May overfit with too many weak base estimators",
                "Weights optimization can be computationally expensive",
                "Performance ceiling limited by base estimator quality"
            ],
            "ideal_use_cases": [
                "When you have multiple good but different algorithms",
                "Projects requiring robust and stable predictions",
                "Competitions where ensemble methods dominate",
                "When individual models have complementary strengths",
                "Applications where slight accuracy improvement is valuable",
                "When you want to reduce model variance",
                "Situations with limited labeled data per algorithm",
                "Problems where different algorithms excel on different regions",
                "When model interpretability can be sacrificed for performance",
                "Applications requiring confidence in predictions"
            ],
            "ensemble_theory": {
                "mathematical_foundation": {
                    "condorcet_jury_theorem": "If each base classifier has accuracy > 0.5, ensemble accuracy approaches 1",
                    "bias_variance_decomposition": "Ensemble error = bias² + variance + noise",
                    "diversity_error_relationship": "Ensemble error decreases with base learner diversity",
                    "optimal_weights": "Can be derived from cross-validation performance"
                },
                "performance_conditions": {
                    "base_accuracy": "Each base learner should be better than random",
                    "diversity_requirement": "Base learners should make different types of errors",
                    "independence_assumption": "Errors should be uncorrelated across base learners",
                    "probability_calibration": "For soft voting, probabilities should be well-calibrated"
                }
            },
            "comparison_with_other_ensembles": {
                "vs_bagging": {
                    "voting": "Different algorithms on same data",
                    "bagging": "Same algorithm on different data subsets",
                    "diversity_source": "Voting: algorithm diversity, Bagging: data diversity"
                },
                "vs_boosting": {
                    "voting": "Independent training of base learners",
                    "boosting": "Sequential training with error correction",
                    "combination": "Voting: equal/weighted vote, Boosting: weighted sequence"
                },
                "vs_stacking": {
                    "voting": "Simple voting/averaging mechanism",
                    "stacking": "Meta-learner learns how to combine base predictions",
                    "complexity": "Voting: simpler, Stacking: more sophisticated"
                }
            },
            "hyperparameter_guide": {
                "voting_strategy": {
                    "hard_voting": "Use when base estimators don't provide good probabilities",
                    "soft_voting": "Generally better if probabilities are well-calibrated",
                    "recommendation": "Try both and use cross-validation to decide"
                },
                "base_estimator_selection": {
                    "minimum_number": "At least 3 for meaningful voting",
                    "maximum_number": "Diminishing returns after 5-7 estimators",
                    "diversity_focus": "Choose algorithms with different assumptions"
                },
                "weight_optimization": {
                    "when_to_use": "When base estimators have significantly different performance",
                    "optimization_metric": "Should match your evaluation metric",
                    "cross_validation": "Use separate validation set to avoid overfitting"
                }
            }
        }
    
    def _create_base_estimators(self) -> List[Tuple[str, BaseEstimator]]:
        """Create base estimators based on configuration"""
        estimators = []
        
        # Logistic Regression
        if self.use_logistic_regression:
            lr = LogisticRegression(
                C=self.lr_C,
                max_iter=self.lr_max_iter,
                random_state=self.random_state
            )
            estimators.append(('logistic_regression', lr))
        
        # Random Forest
        if self.use_random_forest:
            rf = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                random_state=self.rf_random_state
            )
            estimators.append(('random_forest', rf))
        
        # SVM
        if self.use_svm:
            svm = SVC(
                C=self.svm_C,
                kernel=self.svm_kernel,
                probability=self.svm_probability,
                random_state=self.random_state
            )
            estimators.append(('svm', svm))
        
        # K-Nearest Neighbors
        if self.use_knn:
            knn = KNeighborsClassifier(
                n_neighbors=self.knn_n_neighbors,
                weights=self.knn_weights
            )
            estimators.append(('knn', knn))
        
        # Naive Bayes
        if self.use_naive_bayes:
            nb = GaussianNB()
            estimators.append(('naive_bayes', nb))
        
        # Decision Tree
        if self.use_decision_tree:
            dt = DecisionTreeClassifier(
                max_depth=self.dt_max_depth,
                random_state=self.dt_random_state
            )
            estimators.append(('decision_tree', dt))
        
        # Extended algorithms (if available)
        if EXTENDED_ALGORITHMS:
            # Multi-layer Perceptron
            if self.use_mlp:
                mlp = MLPClassifier(
                    hidden_layer_sizes=self.mlp_hidden_layer_sizes,
                    max_iter=self.mlp_max_iter,
                    random_state=self.random_state
                )
                estimators.append(('mlp', mlp))
            
            # Gradient Boosting
            if self.use_gradient_boosting:
                gb = GradientBoostingClassifier(
                    n_estimators=self.gb_n_estimators,
                    learning_rate=self.gb_learning_rate,
                    random_state=self.random_state
                )
                estimators.append(('gradient_boosting', gb))
            
            # AdaBoost
            if self.use_ada_boost:
                ada = AdaBoostClassifier(
                    n_estimators=self.ada_n_estimators,
                    learning_rate=self.ada_learning_rate,
                    random_state=self.random_state
                )
                estimators.append(('ada_boost', ada))
            
            # Linear Discriminant Analysis
            if self.use_lda:
                lda = LinearDiscriminantAnalysis()
                estimators.append(('lda', lda))
            
            # Quadratic Discriminant Analysis
            if self.use_qda:
                qda = QuadraticDiscriminantAnalysis()
                estimators.append(('qda', qda))
        
        if not estimators:
            # Default fallback
            estimators = [
                ('logistic_regression', LogisticRegression(random_state=self.random_state)),
                ('random_forest', RandomForestClassifier(random_state=self.random_state))
            ]
            warnings.warn("No base estimators selected. Using default: Logistic Regression + Random Forest")
        
        return estimators
    
    def _optimize_weights(self, X, y, estimators):
        """Optimize weights for base estimators using cross-validation"""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, f1_score, roc_auc_score
        
        # Define scoring metric
        if self.weight_optimization_method == 'accuracy':
            scoring = 'accuracy'
        elif self.weight_optimization_method == 'f1':
            scoring = make_scorer(f1_score, average='weighted')
        elif self.weight_optimization_method == 'roc_auc':
            if len(self.classes_) == 2:
                scoring = 'roc_auc'
            else:
                scoring = make_scorer(roc_auc_score, average='weighted', multi_class='ovr')
        else:
            scoring = 'accuracy'
        
        # Get cross-validation scores for each base estimator
        base_scores = []
        for name, estimator in estimators:
            try:
                scores = cross_val_score(
                    estimator, X, y,
                    cv=self.cross_validation_folds,
                    scoring=scoring,
                    n_jobs=self.n_jobs
                )
                mean_score = np.mean(scores)
                base_scores.append(mean_score)
                
                if self.verbose:
                    print(f"{name}: {mean_score:.4f} ± {np.std(scores):.4f}")
                    
            except Exception as e:
                warnings.warn(f"Cross-validation failed for {name}: {str(e)}")
                base_scores.append(0.5)  # Default score
        
        # Convert scores to weights (higher score = higher weight)
        base_scores = np.array(base_scores)
        
        # Avoid division by zero
        if np.sum(base_scores) == 0:
            weights = np.ones(len(base_scores)) / len(base_scores)
        else:
            # Normalize scores to get weights
            weights = base_scores / np.sum(base_scores)
        
        # Store individual scores for analysis
        self.base_estimator_scores_ = dict(zip([name for name, _ in estimators], base_scores))
        
        return weights
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Voting Classifier
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, default=None
            Sample weights
            
        Returns:
        --------
        self : object
        """
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
        
        # Feature scaling if requested
        if self.auto_scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
        
        if self.verbose:
            print(f"Training Voting Classifier with {self.voting} voting...")
            print(f"Number of classes: {len(self.classes_)}")
            print(f"Feature scaling: {self.auto_scale_features}")
        
        # Create base estimators
        self.base_estimators_ = self._create_base_estimators()
        
        if self.verbose:
            print(f"Base estimators: {[name for name, _ in self.base_estimators_]}")
        
        # Optimize weights if requested
        if self.optimize_weights:
            if self.verbose:
                print("Optimizing estimator weights using cross-validation...")
            
            self.optimized_weights_ = self._optimize_weights(X_scaled, y_encoded, self.base_estimators_)
            weights = self.optimized_weights_
            
            if self.verbose:
                for i, (name, _) in enumerate(self.base_estimators_):
                    print(f"{name} weight: {weights[i]:.4f}")
        else:
            weights = self.weights
            self.optimized_weights_ = weights
        
        # Create and fit voting classifier
        self.voting_classifier_ = VotingClassifier(
            estimators=self.base_estimators_,
            voting=self.voting,
            weights=weights,
            n_jobs=self.n_jobs,
            flatten_transform=self.flatten_transform,
            verbose=self.verbose
        )
        
        # Fit the ensemble
        self.voting_classifier_.fit(X_scaled, y_encoded, sample_weight=sample_weight)
        
        # Analyze the ensemble
        self._analyze_ensemble(X_scaled, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def _analyze_ensemble(self, X, y):
        """Analyze the trained ensemble"""
        from sklearn.metrics import accuracy_score
        
        analysis = {
            "n_base_estimators": len(self.base_estimators_),
            "voting_strategy": self.voting,
            "feature_scaling": self.scaler_ is not None,
            "weight_optimization": self.optimize_weights,
            "base_estimator_names": [name for name, _ in self.base_estimators_]
        }
        
        # Add weight information
        if self.optimized_weights_ is not None:
            analysis["optimized_weights"] = dict(zip(
                [name for name, _ in self.base_estimators_],
                self.optimized_weights_
            ))
        
        # Add base estimator scores if available
        if self.base_estimator_scores_ is not None:
            analysis["base_estimator_cv_scores"] = self.base_estimator_scores_
        
        # Estimate ensemble performance
        try:
            ensemble_pred = self.voting_classifier_.predict(X)
            ensemble_accuracy = accuracy_score(y, ensemble_pred)
            analysis["ensemble_training_accuracy"] = ensemble_accuracy
        except Exception as e:
            analysis["ensemble_training_accuracy"] = None
        
        self.estimator_analysis_ = analysis
    
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
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get predictions from voting classifier
        y_pred_encoded = self.voting_classifier_.predict(X_scaled)
        
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
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Check if voting classifier supports predict_proba
        if hasattr(self.voting_classifier_, 'predict_proba'):
            try:
                probabilities = self.voting_classifier_.predict_proba(X_scaled)
                return probabilities
            except Exception as e:
                warnings.warn(f"Probability prediction failed: {str(e)}. Using hard predictions.")
        
        # Fallback: convert hard predictions to probabilities
        y_pred = self.voting_classifier_.predict(X_scaled)
        n_samples = len(y_pred)
        n_classes = len(self.classes_)
        
        probabilities = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(y_pred):
            probabilities[i, pred] = 1.0
        
        return probabilities
    
    def get_base_estimator_predictions(self, X):
        """
        Get predictions from individual base estimators
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        predictions : dict
            Dictionary mapping estimator names to their predictions
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        predictions = {}
        
        for name, estimator in self.voting_classifier_.named_estimators_.items():
            try:
                pred_encoded = estimator.predict(X_scaled)
                pred = self.label_encoder_.inverse_transform(pred_encoded)
                predictions[name] = pred
            except Exception as e:
                warnings.warn(f"Prediction failed for {name}: {str(e)}")
                predictions[name] = np.full(X.shape[0], self.classes_[0])
        
        return predictions
    
    def get_ensemble_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of the ensemble
        
        Returns:
        --------
        analysis_info : dict
            Comprehensive ensemble analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "ensemble_summary": {
                "algorithm": "Voting Classifier",
                "voting_strategy": self.voting,
                "n_base_estimators": len(self.base_estimators_),
                "feature_scaling": self.scaler_ is not None,
                "weight_optimization": self.optimize_weights,
                "n_features": self.n_features_in_,
                "n_classes": len(self.classes_),
                "classes": self.classes_.tolist()
            }
        }
        
        # Add base estimator information
        if self.base_estimators_:
            analysis["base_estimators"] = {}
            for name, estimator in self.base_estimators_:
                estimator_info = {
                    "type": type(estimator).__name__,
                    "parameters": estimator.get_params()
                }
                
                # Add weight if available
                if self.optimized_weights_ is not None:
                    idx = [n for n, _ in self.base_estimators_].index(name)
                    estimator_info["weight"] = self.optimized_weights_[idx]
                
                # Add CV score if available
                if self.base_estimator_scores_ and name in self.base_estimator_scores_:
                    estimator_info["cv_score"] = self.base_estimator_scores_[name]
                
                analysis["base_estimators"][name] = estimator_info
        
        # Add ensemble analysis
        if self.estimator_analysis_:
            analysis["ensemble_analysis"] = self.estimator_analysis_
        
        return analysis
    
    def plot_ensemble_analysis(self, figsize=(15, 10)):
        """
        Create comprehensive ensemble analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Ensemble analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Base Estimator Weights
        if self.optimized_weights_ is not None:
            estimator_names = [name for name, _ in self.base_estimators_]
            weights = self.optimized_weights_
            
            bars = ax1.bar(estimator_names, weights, alpha=0.7, color='skyblue', edgecolor='navy')
            ax1.set_xlabel('Base Estimators')
            ax1.set_ylabel('Weight')
            ax1.set_title('Base Estimator Weights in Ensemble')
            ax1.tick_params(axis='x', rotation=45)
            
            # Highlight most important estimator
            max_weight_idx = np.argmax(weights)
            bars[max_weight_idx].set_color('orange')
            
            # Add weight values on bars
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{weight:.3f}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'Equal weights\n(no optimization)', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Base Estimator Weights')
        
        # 2. Base Estimator CV Scores
        if self.base_estimator_scores_:
            names = list(self.base_estimator_scores_.keys())
            scores = list(self.base_estimator_scores_.values())
            
            bars = ax2.bar(names, scores, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
            ax2.set_xlabel('Base Estimators')
            ax2.set_ylabel('CV Score')
            ax2.set_title('Base Estimator Cross-Validation Performance')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylim(0, 1)
            
            # Highlight best performer
            max_score_idx = np.argmax(scores)
            bars[max_score_idx].set_color('gold')
            
            # Add score values on bars
            for bar, score in zip(bars, scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'CV scores\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Base Estimator CV Scores')
        
        # 3. Voting Strategy Visualization
        voting_info = {
            'Strategy': [self.voting.capitalize()],
            'Description': ['Probability averaging' if self.voting == 'soft' else 'Majority voting']
        }
        
        ax3.axis('tight')
        ax3.axis('off')
        table_data = []
        table_data.append(['Voting Strategy', self.voting.capitalize()])
        table_data.append(['Number of Estimators', str(len(self.base_estimators_))])
        table_data.append(['Weight Optimization', 'Yes' if self.optimize_weights else 'No'])
        table_data.append(['Feature Scaling', 'Yes' if self.scaler_ is not None else 'No'])
        
        table = ax3.table(cellText=table_data,
                         colLabels=['Property', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax3.set_title('Ensemble Configuration')
        
        # 4. Estimator Type Distribution
        estimator_types = {}
        for name, estimator in self.base_estimators_:
            estimator_type = type(estimator).__name__
            estimator_types[estimator_type] = estimator_types.get(estimator_type, 0) + 1
        
        if estimator_types:
            types = list(estimator_types.keys())
            counts = list(estimator_types.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            wedges, texts, autotexts = ax4.pie(counts, labels=types, autopct='%1.0f%%',
                                              colors=colors, startangle=90)
            ax4.set_title('Base Estimator Type Distribution')
        else:
            ax4.text(0.5, 0.5, 'No estimator\ntype data', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Estimator Types')
        
        plt.tight_layout()
        return fig
    
    def plot_voting_comparison(self, X_test, y_test, figsize=(12, 8)):
        """
        Compare individual estimator predictions with ensemble prediction
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Voting comparison plot
        """
        if not self.is_fitted_:
            return None
        
        # Get predictions from all estimators
        base_predictions = self.get_base_estimator_predictions(X_test)
        ensemble_prediction = self.predict(X_test)
        
        # Calculate accuracy for each estimator
        from sklearn.metrics import accuracy_score
        
        estimator_names = list(base_predictions.keys()) + ['Ensemble']
        accuracies = []
        
        # Base estimator accuracies
        for name, pred in base_predictions.items():
            acc = accuracy_score(y_test, pred)
            accuracies.append(acc)
        
        # Ensemble accuracy
        ensemble_acc = accuracy_score(y_test, ensemble_prediction)
        accuracies.append(ensemble_acc)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Accuracy Comparison
        colors = ['skyblue'] * len(base_predictions) + ['orange']
        bars = ax1.bar(estimator_names, accuracies, color=colors, alpha=0.7, edgecolor='navy')
        ax1.set_xlabel('Estimators')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Individual vs Ensemble Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        
        # Highlight ensemble
        bars[-1].set_edgecolor('red')
        bars[-1].set_linewidth(2)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Prediction Agreement Analysis
        # Count how often each estimator agrees with ensemble
        agreement_counts = []
        for name, pred in base_predictions.items():
            agreement = np.sum(pred == ensemble_prediction) / len(pred)
            agreement_counts.append(agreement)
        
        bars2 = ax2.bar(list(base_predictions.keys()), agreement_counts, 
                       alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.set_xlabel('Base Estimators')
        ax2.set_ylabel('Agreement with Ensemble')
        ax2.set_title('Base Estimator Agreement with Ensemble')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add agreement values on bars
        for bar, agreement in zip(bars2, agreement_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{agreement:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🗳️ Voting Classifier Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Voting", "Estimators", "Hyperparams", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Voting Strategy**")
            
            # Voting strategy
            voting = st.selectbox(
                "Voting Method:",
                options=['soft', 'hard'],
                index=['soft', 'hard'].index(self.voting),
                help="soft: average probabilities, hard: majority vote",
                key=f"{key_prefix}_voting"
            )
            
            if voting == 'soft':
                st.info("🎯 Soft Voting: Averages predicted probabilities for smoother decisions")
            else:
                st.info("🗳️ Hard Voting: Uses majority vote from class predictions")
            
            # Weight optimization
            optimize_weights = st.checkbox(
                "Optimize Estimator Weights",
                value=self.optimize_weights,
                help="Use cross-validation to optimize weights for base estimators",
                key=f"{key_prefix}_optimize_weights"
            )
            
            if optimize_weights:
                weight_optimization_method = st.selectbox(
                    "Weight Optimization Metric:",
                    options=['accuracy', 'f1', 'roc_auc'],
                    index=['accuracy', 'f1', 'roc_auc'].index(self.weight_optimization_method),
                    help="Metric for optimizing estimator weights",
                    key=f"{key_prefix}_weight_optimization_method"
                )
                
                cross_validation_folds = st.slider(
                    "CV Folds for Weight Optimization:",
                    min_value=3,
                    max_value=10,
                    value=self.cross_validation_folds,
                    help="Number of cross-validation folds",
                    key=f"{key_prefix}_cross_validation_folds"
                )
            else:
                weight_optimization_method = self.weight_optimization_method
                cross_validation_folds = self.cross_validation_folds
            
            # Feature scaling
            auto_scale_features = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scale_features,
                help="Automatically scale features (recommended when using diverse algorithms)",
                key=f"{key_prefix}_auto_scale_features"
            )
        
        with tab2:
            st.markdown("**Base Estimator Selection**")
            
            # Core estimators
            st.markdown("**Core Algorithms:**")
            use_logistic_regression = st.checkbox(
                "Logistic Regression",
                value=self.use_logistic_regression,
                help="Fast linear classifier with good probabilities",
                key=f"{key_prefix}_use_logistic_regression"
            )
            
            use_random_forest = st.checkbox(
                "Random Forest",
                value=self.use_random_forest,
                help="Robust tree ensemble, handles non-linearity well",
                key=f"{key_prefix}_use_random_forest"
            )
            
            use_svm = st.checkbox(
                "Support Vector Machine",
                value=self.use_svm,
                help="Kernel-based classifier, good for complex boundaries",
                key=f"{key_prefix}_use_svm"
            )
            
            use_knn = st.checkbox(
                "K-Nearest Neighbors",
                value=self.use_knn,
                help="Instance-based learning, captures local patterns",
                key=f"{key_prefix}_use_knn"
            )
            
            use_naive_bayes = st.checkbox(
                "Naive Bayes",
                value=self.use_naive_bayes,
                help="Fast probabilistic classifier, works well with small data",
                key=f"{key_prefix}_use_naive_bayes"
            )
            
            # Optional estimators
            st.markdown("**Additional Algorithms:**")
            use_decision_tree = st.checkbox(
                "Decision Tree",
                value=self.use_decision_tree,
                help="Interpretable tree-based classifier",
                key=f"{key_prefix}_use_decision_tree"
            )
            
            if EXTENDED_ALGORITHMS:
                use_mlp = st.checkbox(
                    "Neural Network (MLP)",
                    value=self.use_mlp,
                    help="Multi-layer perceptron for complex patterns",
                    key=f"{key_prefix}_use_mlp"
                )
                
                use_gradient_boosting = st.checkbox(
                    "Gradient Boosting",
                    value=self.use_gradient_boosting,
                    help="Sequential boosting for high performance",
                    key=f"{key_prefix}_use_gradient_boosting"
                )
                
                use_ada_boost = st.checkbox(
                    "AdaBoost",
                    value=self.use_ada_boost,
                    help="Adaptive boosting algorithm",
                    key=f"{key_prefix}_use_ada_boost"
                )
                
                use_lda = st.checkbox(
                    "Linear Discriminant Analysis",
                    value=self.use_lda,
                    help="Linear dimensionality reduction classifier",
                    key=f"{key_prefix}_use_lda"
                )
                
                use_qda = st.checkbox(
                    "Quadratic Discriminant Analysis",
                    value=self.use_qda,
                    help="Quadratic decision boundaries",
                    key=f"{key_prefix}_use_qda"
                )
            else:
                use_mlp = self.use_mlp
                use_gradient_boosting = self.use_gradient_boosting
                use_ada_boost = self.use_ada_boost
                use_lda = self.use_lda
                use_qda = self.use_qda
            
            # Count selected estimators
            selected_count = sum([
                use_logistic_regression, use_random_forest, use_svm, use_knn,
                use_naive_bayes, use_decision_tree, use_mlp, use_gradient_boosting,
                use_ada_boost, use_lda, use_qda
            ])
            
            if selected_count < 2:
                st.warning("⚠️ Select at least 2 estimators for meaningful voting!")
            elif selected_count >= 5:
                st.info(f"✅ {selected_count} estimators selected - good diversity!")
            else:
                st.success(f"✅ {selected_count} estimators selected")
        
        with tab3:
            st.markdown("**Base Estimator Hyperparameters**")
            
            # Logistic Regression parameters
            if use_logistic_regression:
                st.markdown("**Logistic Regression:**")
                lr_C = st.number_input(
                    "LR Regularization (C):",
                    value=float(self.lr_C),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_lr_C"
                )
                
                lr_max_iter = st.number_input(
                    "LR Max Iterations:",
                    value=int(self.lr_max_iter),
                    min_value=100,
                    max_value=5000,
                    step=100,
                    key=f"{key_prefix}_lr_max_iter"
                )
            else:
                lr_C = self.lr_C
                lr_max_iter = self.lr_max_iter
            
            # Random Forest parameters
            if use_random_forest:
                st.markdown("**Random Forest:**")
                rf_n_estimators = st.slider(
                    "RF Number of Trees:",
                    min_value=10,
                    max_value=500,
                    value=int(self.rf_n_estimators),
                    step=10,
                    key=f"{key_prefix}_rf_n_estimators"
                )
                
                rf_max_depth_option = st.selectbox(
                    "RF Max Depth:",
                    options=['None', 'Custom'],
                    index=0 if self.rf_max_depth is None else 1,
                    key=f"{key_prefix}_rf_max_depth_option"
                )
                
                if rf_max_depth_option == 'Custom':
                    rf_max_depth = st.slider(
                        "RF Custom Max Depth:",
                        min_value=1,
                        max_value=50,
                        value=10 if self.rf_max_depth is None else int(self.rf_max_depth),
                        key=f"{key_prefix}_rf_max_depth_custom"
                    )
                else:
                    rf_max_depth = None
            else:
                rf_n_estimators = self.rf_n_estimators
                rf_max_depth = self.rf_max_depth
            
            # SVM parameters
            if use_svm:
                st.markdown("**Support Vector Machine:**")
                svm_C = st.number_input(
                    "SVM Regularization (C):",
                    value=float(self.svm_C),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_svm_C"
                )
                
                svm_kernel = st.selectbox(
                    "SVM Kernel:",
                    options=['rbf', 'linear', 'poly', 'sigmoid'],
                    index=['rbf', 'linear', 'poly', 'sigmoid'].index(self.svm_kernel),
                    key=f"{key_prefix}_svm_kernel"
                )
            else:
                svm_C = self.svm_C
                svm_kernel = self.svm_kernel
            
            # KNN parameters
            if use_knn:
                st.markdown("**K-Nearest Neighbors:**")
                knn_n_neighbors = st.slider(
                    "KNN Number of Neighbors:",
                    min_value=1,
                    max_value=50,
                    value=int(self.knn_n_neighbors),
                    key=f"{key_prefix}_knn_n_neighbors"
                )
                
                knn_weights = st.selectbox(
                    "KNN Weights:",
                    options=['uniform', 'distance'],
                    index=['uniform', 'distance'].index(self.knn_weights),
                    key=f"{key_prefix}_knn_weights"
                )
            else:
                knn_n_neighbors = self.knn_n_neighbors
                knn_weights = self.knn_weights
        
        with tab4:
            st.markdown("**Advanced Settings**")
            
            # Neural Network parameters (if selected and available)
            if use_mlp and EXTENDED_ALGORITHMS:
                st.markdown("**Neural Network:**")
                mlp_hidden_layer_sizes = st.text_input(
                    "MLP Hidden Layers (comma-separated):",
                    value=','.join(map(str, self.mlp_hidden_layer_sizes)),
                    help="e.g., '100' or '100,50' for multiple layers",
                    key=f"{key_prefix}_mlp_hidden_layer_sizes"
                )
                
                try:
                    mlp_hidden_layer_sizes = tuple(map(int, mlp_hidden_layer_sizes.split(',')))
                except:
                    mlp_hidden_layer_sizes = self.mlp_hidden_layer_sizes
                
                mlp_max_iter = st.number_input(
                    "MLP Max Iterations:",
                    value=int(self.mlp_max_iter),
                    min_value=100,
                    max_value=2000,
                    step=100,
                    key=f"{key_prefix}_mlp_max_iter"
                )
            else:
                mlp_hidden_layer_sizes = self.mlp_hidden_layer_sizes
                mlp_max_iter = self.mlp_max_iter
            
            # Gradient Boosting parameters (if selected and available)
            if use_gradient_boosting and EXTENDED_ALGORITHMS:
                st.markdown("**Gradient Boosting:**")
                gb_n_estimators = st.slider(
                    "GB Number of Estimators:",
                    min_value=10,
                    max_value=300,
                    value=int(self.gb_n_estimators),
                    step=10,
                    key=f"{key_prefix}_gb_n_estimators"
                )
                
                gb_learning_rate = st.number_input(
                    "GB Learning Rate:",
                    value=float(self.gb_learning_rate),
                    min_value=0.01,
                    max_value=1.0,
                    step=0.01,
                    format="%.3f",
                    key=f"{key_prefix}_gb_learning_rate"
                )
            else:
                gb_n_estimators = self.gb_n_estimators
                gb_learning_rate = self.gb_learning_rate
            
            # Parallel processing
            n_jobs = st.selectbox(
                "Parallel Jobs:",
                options=[None, 1, 2, 4, -1],
                index=0,
                help="-1 uses all available cores",
                key=f"{key_prefix}_n_jobs"
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
            
            # Verbose output
            verbose = st.checkbox(
                "Verbose Output",
                value=self.verbose,
                help="Print training progress and details",
                key=f"{key_prefix}_verbose"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Voting Classifier** - Ensemble Meta-Algorithm:
            • 🗳️ Combines multiple diverse algorithms
            • 🎯 Two voting strategies: hard (majority) and soft (probability averaging)
            • 📊 Often outperforms individual base estimators
            • 🔄 Reduces overfitting through model diversity
            • ⚖️ Can optimize weights for base estimators
            • 🚀 Leverages strengths of different algorithm types
            
            **Key Principles:**
            • Diversity among base estimators is crucial
            • Soft voting generally performs better
            • Weight optimization can improve performance
            • Feature scaling helps with diverse algorithms
            """)
            
            # Algorithm selection guide
            if st.button("🎯 Estimator Selection Guide", key=f"{key_prefix}_selection_guide"):
                st.markdown("""
                **Recommended Base Estimator Combinations:**
                
                **Balanced Ensemble (Recommended):**
                - Logistic Regression (linear, fast)
                - Random Forest (non-linear, robust)
                - SVM (kernel-based, flexible)
                - K-NN (instance-based, local)
                
                **High Performance Ensemble:**
                - Random Forest (tree ensemble)
                - Gradient Boosting (sequential boosting)
                - SVM with RBF kernel (non-linear)
                - Neural Network (deep patterns)
                
                **Fast & Simple Ensemble:**
                - Logistic Regression (linear)
                - Naive Bayes (probabilistic)
                - K-NN (instance-based)
                
                **Diversity Principles:**
                - Mix linear and non-linear algorithms
                - Combine parametric and non-parametric methods
                - Include both global and local learners
                - Balance fast and complex algorithms
                """)
            
            # Voting strategy guide
            if st.button("🗳️ Voting Strategy Guide", key=f"{key_prefix}_voting_guide"):
                st.markdown("""
                **Voting Strategy Selection:**
                
                **Soft Voting (Recommended):**
                - Uses probability estimates from base classifiers
                - Generally provides better performance
                - Creates smoother decision boundaries
                - Requires base estimators to support predict_proba()
                - Best when probabilities are well-calibrated
                
                **Hard Voting:**
                - Uses class predictions (majority vote)
                - Simpler and more interpretable
                - Works with any classifier
                - Faster prediction time
                - Best when probability estimates are unreliable
                
                **Weight Optimization:**
                - Automatically determines optimal weights
                - Based on cross-validation performance
                - Can significantly improve ensemble performance
                - Adds computational overhead
                - Recommended for competitive performance
                """)
            
            # Performance tuning guide
            if st.button("⚙️ Performance Tuning Guide", key=f"{key_prefix}_tuning_guide"):
                st.markdown("""
                **Voting Classifier Optimization:**
                
                **Base Estimator Selection:**
                - Start with 3-5 diverse algorithms
                - Ensure each estimator performs reasonably well
                - Remove very poor performing estimators
                - Add complexity gradually
                
                # Continue from line 1505 where the code breaks:
                
                **Weight Optimization:**
                - Enable weight optimization for better performance
                - Choose metric that matches your evaluation criteria
                - Use 5-fold CV for reliable weight estimation
                - Monitor individual estimator contributions
                
                **Feature Scaling:**
                - Enable when using diverse algorithms
                - Especially important with SVM and KNN
                - Less critical if using only tree-based methods
                - StandardScaler generally works well
                
                **Hyperparameter Tuning:**
                - Tune base estimators individually first
                - Then optimize ensemble-level parameters
                - Focus on diversity over individual performance
                - Use grid search for systematic optimization
                """)
            
            # Ensemble theory
            if st.button("📚 Ensemble Theory", key=f"{key_prefix}_ensemble_theory"):
                st.markdown("""
                **Mathematical Foundation:**
                
                **Condorcet's Jury Theorem:**
                - If each classifier has accuracy > 0.5
                - Majority vote accuracy approaches 1 as n → ∞
                - Requires error independence assumption
                
                **Bias-Variance Decomposition:**
                - Ensemble Error = Bias² + Variance + Noise
                - Voting reduces variance through averaging
                - Bias remains similar to base estimators
                - Net effect: improved generalization
                
                **Diversity-Accuracy Tradeoff:**
                - More diverse estimators → better ensemble
                - But individual accuracy shouldn't be too low
                - Optimal balance depends on problem
                - Weight optimization helps find this balance
                
                **Soft vs Hard Voting:**
                - Soft: P(y|x) = Σ w_i * P_i(y|x)
                - Hard: y = argmax_c Σ w_i * I(h_i(x) = c)
                - Soft generally superior with calibrated probabilities
                """)
        
        return {
            "voting": voting,
            "optimize_weights": optimize_weights,
            "weight_optimization_method": weight_optimization_method,
            "cross_validation_folds": cross_validation_folds,
            "auto_scale_features": auto_scale_features,
            "use_logistic_regression": use_logistic_regression,
            "use_random_forest": use_random_forest,
            "use_svm": use_svm,
            "use_knn": use_knn,
            "use_naive_bayes": use_naive_bayes,
            "use_decision_tree": use_decision_tree,
            "use_mlp": use_mlp,
            "use_gradient_boosting": use_gradient_boosting,
            "use_ada_boost": use_ada_boost,
            "use_lda": use_lda,
            "use_qda": use_qda,
            "lr_C": lr_C,
            "lr_max_iter": lr_max_iter,
            "rf_n_estimators": rf_n_estimators,
            "rf_max_depth": rf_max_depth,
            "svm_C": svm_C,
            "svm_kernel": svm_kernel,
            "knn_n_neighbors": knn_n_neighbors,
            "knn_weights": knn_weights,
            "mlp_hidden_layer_sizes": mlp_hidden_layer_sizes,
            "mlp_max_iter": mlp_max_iter,
            "gb_n_estimators": gb_n_estimators,
            "gb_learning_rate": gb_learning_rate,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return VotingClassifierPlugin(
            voting=hyperparameters.get("voting", self.voting),
            optimize_weights=hyperparameters.get("optimize_weights", self.optimize_weights),
            weight_optimization_method=hyperparameters.get("weight_optimization_method", self.weight_optimization_method),
            cross_validation_folds=hyperparameters.get("cross_validation_folds", self.cross_validation_folds),
            auto_scale_features=hyperparameters.get("auto_scale_features", self.auto_scale_features),
            use_logistic_regression=hyperparameters.get("use_logistic_regression", self.use_logistic_regression),
            use_random_forest=hyperparameters.get("use_random_forest", self.use_random_forest),
            use_svm=hyperparameters.get("use_svm", self.use_svm),
            use_knn=hyperparameters.get("use_knn", self.use_knn),
            use_naive_bayes=hyperparameters.get("use_naive_bayes", self.use_naive_bayes),
            use_decision_tree=hyperparameters.get("use_decision_tree", self.use_decision_tree),
            use_mlp=hyperparameters.get("use_mlp", self.use_mlp),
            use_gradient_boosting=hyperparameters.get("use_gradient_boosting", self.use_gradient_boosting),
            use_ada_boost=hyperparameters.get("use_ada_boost", self.use_ada_boost),
            use_lda=hyperparameters.get("use_lda", self.use_lda),
            use_qda=hyperparameters.get("use_qda", self.use_qda),
            lr_C=hyperparameters.get("lr_C", self.lr_C),
            lr_max_iter=hyperparameters.get("lr_max_iter", self.lr_max_iter),
            rf_n_estimators=hyperparameters.get("rf_n_estimators", self.rf_n_estimators),
            rf_max_depth=hyperparameters.get("rf_max_depth", self.rf_max_depth),
            svm_C=hyperparameters.get("svm_C", self.svm_C),
            svm_kernel=hyperparameters.get("svm_kernel", self.svm_kernel),
            knn_n_neighbors=hyperparameters.get("knn_n_neighbors", self.knn_n_neighbors),
            knn_weights=hyperparameters.get("knn_weights", self.knn_weights),
            mlp_hidden_layer_sizes=hyperparameters.get("mlp_hidden_layer_sizes", self.mlp_hidden_layer_sizes),
            mlp_max_iter=hyperparameters.get("mlp_max_iter", self.mlp_max_iter),
            gb_n_estimators=hyperparameters.get("gb_n_estimators", self.gb_n_estimators),
            gb_learning_rate=hyperparameters.get("gb_learning_rate", self.gb_learning_rate),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            random_state=hyperparameters.get("random_state", self.random_state),
            verbose=hyperparameters.get("verbose", self.verbose)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for Voting Classifier
        
        The voting classifier can handle various data types depending on base estimators,
        but feature scaling is recommended when using diverse algorithms.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # Check for missing values
        if np.any(pd.isna(X_processed)):
            warnings.warn("Some base estimators don't handle missing values well. Consider imputation.")
        
        # Check for infinite values
        if np.any(np.isinf(X_processed)):
            warnings.warn("Infinite values detected. Some base estimators may fail.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if Voting Classifier is compatible with the given data
        
        Returns:
        --------
        compatible : bool
            Whether the algorithm is compatible
        message : str
            Explanation message
        """
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Voting Classifier requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check if we have at least 2 base estimators selected
        selected_estimators = sum([
            self.use_logistic_regression,
            self.use_random_forest,
            self.use_svm,
            self.use_knn,
            self.use_naive_bayes,
            self.use_decision_tree,
            self.use_mlp,
            self.use_gradient_boosting,
            self.use_ada_boost,
            self.use_lda,
            self.use_qda
        ])
        
        if selected_estimators < 2:
            return False, "Voting Classifier needs at least 2 base estimators. Please select more algorithms."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            # Check class distribution
            class_counts = np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes)
            min_class_size = np.min(class_counts)
            if min_class_size < 5:
                return True, f"Warning: Very small class detected ({min_class_size} samples). May affect some base estimators."
        
        # Check for missing values
        if np.any(pd.isna(X)):
            return True, "Warning: Missing values detected. Some base estimators may require imputation."
        
        # Check dimensionality for specific estimators
        if X.shape[1] > 1000:
            warning_msg = "High dimensionality detected. Consider feature selection, especially for SVM and KNN."
            if self.use_svm or self.use_knn:
                return True, warning_msg
        
        return True, f"Voting Classifier is compatible! Using {selected_estimators} base estimators with {self.voting} voting."
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "voting_strategy": self.voting,
            "n_base_estimators": len(self.base_estimators_) if self.base_estimators_ else None,
            "base_estimator_names": [name for name, _ in self.base_estimators_] if self.base_estimators_ else None,
            "weight_optimization": self.optimize_weights,
            "optimized_weights": self.optimized_weights_,
            "feature_scaling": self.scaler_ is not None,
            "ensemble_type": "Voting Classifier"
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Voting Classifier",
            "ensemble_type": "Voting Ensemble",
            "training_completed": True,
            "voting_characteristics": {
                "meta_algorithm": True,
                "ensemble_method": True,
                "combines_diverse_models": True,
                "voting_based": True,
                "probability_averaging": self.voting == 'soft',
                "majority_voting": self.voting == 'hard',
                "weight_optimization": self.optimize_weights
            },
            "ensemble_configuration": {
                "voting_strategy": self.voting,
                "n_base_estimators": len(self.base_estimators_) if self.base_estimators_ else None,
                "base_estimators": [name for name, _ in self.base_estimators_] if self.base_estimators_ else None,
                "feature_scaling": self.scaler_ is not None,
                "weight_optimization": self.optimize_weights,
                "optimization_metric": self.weight_optimization_method if self.optimize_weights else None
            },
            "ensemble_analysis": self.get_ensemble_analysis(),
            "performance_considerations": {
                "training_time": "Moderate - trains multiple base estimators",
                "prediction_time": "Moderate - queries all base estimators",
                "memory_usage": "High - stores multiple models",
                "scalability": "Good - parallelizable across estimators",
                "interpretability": "Moderate - can analyze individual contributions",
                "robustness": "High - ensemble reduces variance"
            },
            "ensemble_theory": {
                "diversity_principle": "Combines different algorithm types for better performance",
                "bias_variance_tradeoff": "Reduces variance through model averaging",
                "condorcet_theorem": "Majority voting improves accuracy if base classifiers > 50% accurate",
                "probability_averaging": "Soft voting provides smoother decision boundaries",
                "weight_optimization": "Can learn optimal combination weights from data"
            }
        }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Voting Classifier model.

        These metrics are derived from the model's learned parameters,
        base estimator evaluations (if weight optimization was performed),
        and ensemble configuration.
        Parameters y_true, y_pred, y_proba are kept for API consistency but are not
        directly used as metrics are sourced from the fitted model's internal attributes.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values.
        y_pred : np.ndarray, optional
            Predicted target values.
        y_proba : np.ndarray, optional
            Predicted probabilities.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not hasattr(self, 'voting_classifier_') or self.voting_classifier_ is None:
            return {"error": "Model not fitted. Cannot retrieve Voting Classifier specific metrics."}

        metrics = {}
        prefix = "voting_"  # Prefix for Voting Classifier specific metrics

        # Ensemble Configuration Metrics
        if self.base_estimators_:
            metrics[f"{prefix}num_base_estimators"] = len(self.base_estimators_)
            
            estimator_types = [type(est).__name__ for _, est in self.base_estimators_]
            metrics[f"{prefix}num_unique_estimator_types"] = len(set(estimator_types))

        metrics[f"{prefix}strategy"] = self.voting
        metrics[f"{prefix}weights_optimized"] = 1 if self.optimize_weights else 0

        # Optimized Weights Statistics (if applicable)
        # self.optimized_weights_ stores the weights used. If optimize_weights was True, these are optimized.
        # If optimize_weights was False, self.optimized_weights_ might be None or user-provided.
        # We are interested in the actual weights used by the classifier.
        actual_weights = self.voting_classifier_.weights
        if actual_weights is not None:
            actual_weights_np = np.array(actual_weights)
            metrics[f"{prefix}mean_actual_weight"] = float(np.mean(actual_weights_np))
            metrics[f"{prefix}std_actual_weight"] = float(np.std(actual_weights_np))
            metrics[f"{prefix}min_actual_weight"] = float(np.min(actual_weights_np))
            metrics[f"{prefix}max_actual_weight"] = float(np.max(actual_weights_np))
            metrics[f"{prefix}num_estimators_with_weights"] = len(actual_weights_np)

        # Base Estimator CV Scores (if weight optimization was performed and scores are stored)
        if self.optimize_weights and hasattr(self, 'base_estimator_scores_') and self.base_estimator_scores_:
            scores = [score for score in self.base_estimator_scores_.values() if isinstance(score, (int, float))]
            if scores:
                metrics[f"{prefix}mean_base_estimator_cv_score"] = float(np.mean(scores))
                metrics[f"{prefix}std_base_estimator_cv_score"] = float(np.std(scores))
                metrics[f"{prefix}min_base_estimator_cv_score"] = float(np.min(scores))
                metrics[f"{prefix}max_base_estimator_cv_score"] = float(np.max(scores))
                metrics[f"{prefix}num_base_estimators_cv_scored"] = len(scores)
        
        # Information from estimator_analysis_ if available
        if hasattr(self, 'estimator_analysis_') and self.estimator_analysis_:
            if "ensemble_training_accuracy" in self.estimator_analysis_ and self.estimator_analysis_["ensemble_training_accuracy"] is not None:
                metrics[f"{prefix}ensemble_training_accuracy"] = float(self.estimator_analysis_["ensemble_training_accuracy"])

        if not metrics:
            metrics['info'] = "No specific Voting Classifier metrics were available from internal analyses."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return VotingClassifierPlugin()