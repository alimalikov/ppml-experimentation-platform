import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Try to import MLP with graceful fallback
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import log_loss
    import scipy.stats as stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    MLPClassifier = None

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

class MLPClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Multi-layer Perceptron Classifier Plugin - Standard Neural Network
    
    MLP is a feedforward artificial neural network that consists of multiple layers
    of nodes in a directed graph, with each layer fully connected to the next one.
    It's the foundation of deep learning and can approximate any continuous function.
    """
    
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 solver='adam',
                 alpha=0.0001,
                 batch_size='auto',
                 learning_rate='constant',
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=42,
                 tol=1e-4,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10,
                 max_fun=15000,
                 # Custom parameters
                 auto_scaling=True,
                 scaling_method='standard',
                 architecture_optimization=False,
                 dropout_simulation=False,
                 custom_architecture=None):
        """
        Initialize MLP Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple, default=(100,)
            Number of neurons in each hidden layer
        activation : str, default='relu'
            Activation function ('identity', 'logistic', 'tanh', 'relu')
        solver : str, default='adam'
            Solver for weight optimization ('lbfgs', 'sgd', 'adam')
        alpha : float, default=0.0001
            L2 penalty (regularization term) parameter
        batch_size : int or 'auto', default='auto'
            Size of minibatches for stochastic optimizers
        learning_rate : str, default='constant'
            Learning rate schedule ('constant', 'invscaling', 'adaptive')
        learning_rate_init : float, default=0.001
            Initial learning rate
        power_t : float, default=0.5
            Exponent for inverse scaling learning rate
        max_iter : int, default=200
            Maximum number of iterations
        shuffle : bool, default=True
            Whether to shuffle samples in each iteration
        random_state : int, default=42
            Random seed for reproducibility
        tol : float, default=1e-4
            Tolerance for optimization
        verbose : bool, default=False
            Whether to print progress messages
        warm_start : bool, default=False
            Reuse solution of previous call to fit
        momentum : float, default=0.9
            Momentum for gradient descent update (only for SGD)
        nesterovs_momentum : bool, default=True
            Whether to use Nesterov's momentum (only for SGD)
        early_stopping : bool, default=False
            Whether to use early stopping to terminate training
        validation_fraction : float, default=0.1
            Fraction of training data for validation
        beta_1 : float, default=0.9
            Exponential decay rate for first moment estimates (Adam)
        beta_2 : float, default=0.999
            Exponential decay rate for second moment estimates (Adam)
        epsilon : float, default=1e-8
            Value for numerical stability (Adam)
        n_iter_no_change : int, default=10
            Maximum number of epochs with no improvement for early stopping
        max_fun : int, default=15000
            Maximum number of loss function calls (LBFGS only)
        auto_scaling : bool, default=True
            Whether to automatically scale features
        scaling_method : str, default='standard'
            Scaling method ('standard', 'minmax')
        architecture_optimization : bool, default=False
            Whether to optimize architecture automatically
        dropout_simulation : bool, default=False
            Simulate dropout for regularization analysis
        custom_architecture : list, optional
            Custom architecture specification
        """
        super().__init__()
        
        # Core MLP parameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        
        # Custom parameters
        self.auto_scaling = auto_scaling
        self.scaling_method = scaling_method
        self.architecture_optimization = architecture_optimization
        self.dropout_simulation = dropout_simulation
        self.custom_architecture = custom_architecture
        
        # Plugin metadata
        self._name = "Multi-layer Perceptron"
        self._description = "Standard neural network with multiple hidden layers for complex pattern recognition."
        self._category = "Neural Networks"
        self._algorithm_type = "Feedforward Neural Network"
        self._paper_reference = "Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 50
        self._handles_missing_values = False
        self._requires_scaling = True  # CRITICAL for neural networks
        self._supports_sparse = False
        self._is_linear = False
        self._provides_feature_importance = False
        self._provides_probabilities = True
        self._handles_categorical = False
        self._ensemble_method = False
        self._iterative_learning = True
        self._gradient_based = True
        self._universal_approximator = True
        self._deep_learning_foundation = True
        self._backpropagation = True
        self._non_convex_optimization = True
        
        # Internal attributes
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.training_history_ = None
        self.learning_curve_data_ = None
        self.architecture_analysis_ = None
        self.convergence_analysis_ = None
        
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
            "year_introduced": 1943,  # McCulloch-Pitts neuron
            "modern_form": 1986,      # Backpropagation
            "key_innovations": {
                "universal_approximation": "Can approximate any continuous function given enough neurons",
                "backpropagation": "Efficient algorithm for training multi-layer networks",
                "non_linear_mapping": "Multiple activation functions enable complex decision boundaries",
                "gradient_descent": "Iterative optimization through gradient-based learning",
                "hidden_representations": "Learns intermediate feature representations",
                "feedforward_architecture": "Information flows from input to output through hidden layers"
            },
            "algorithm_mechanics": {
                "network_structure": {
                    "input_layer": "Receives input features",
                    "hidden_layers": "Apply weighted transformations and non-linear activations", 
                    "output_layer": "Produces final predictions/probabilities",
                    "connections": "Fully connected between adjacent layers"
                },
                "forward_pass": {
                    "step_1": "Input × Weights + Bias",
                    "step_2": "Apply activation function",
                    "step_3": "Repeat for each layer",
                    "step_4": "Output layer produces predictions"
                },
                "backward_pass": {
                    "step_1": "Calculate loss/error",
                    "step_2": "Compute gradients via chain rule",
                    "step_3": "Propagate errors backward",
                    "step_4": "Update weights and biases"
                },
                "optimization_process": {
                    "initialization": "Random weight initialization",
                    "iteration": "Forward pass → Loss calculation → Backward pass → Weight update",
                    "convergence": "Repeat until loss stabilizes or max iterations reached"
                }
            },
            "activation_functions": {
                "relu": {
                    "formula": "f(x) = max(0, x)",
                    "advantages": ["Avoids vanishing gradient", "Computationally efficient", "Sparse activation"],
                    "disadvantages": ["Dying ReLU problem", "Not zero-centered"],
                    "use_case": "Default choice for hidden layers"
                },
                "logistic": {
                    "formula": "f(x) = 1 / (1 + e^(-x))",
                    "advantages": ["Smooth gradient", "Output in (0,1)", "Historically significant"],
                    "disadvantages": ["Vanishing gradient", "Not zero-centered", "Saturating"],
                    "use_case": "Binary classification output, historical networks"
                },
                "tanh": {
                    "formula": "f(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
                    "advantages": ["Zero-centered", "Smooth gradient", "Output in (-1,1)"],
                    "disadvantages": ["Vanishing gradient", "Saturating"],
                    "use_case": "When zero-centered output is preferred"
                },
                "identity": {
                    "formula": "f(x) = x",
                    "advantages": ["No vanishing gradient", "Simple"],
                    "disadvantages": ["Linear only", "No non-linearity"],
                    "use_case": "Output layer for regression, debugging"
                }
            },
            "optimizers": {
                "adam": {
                    "description": "Adaptive Moment Estimation",
                    "advantages": ["Adaptive learning rates", "Works well in practice", "Handles sparse gradients"],
                    "parameters": ["learning_rate", "beta_1", "beta_2", "epsilon"],
                    "use_case": "Default choice for most problems"
                },
                "sgd": {
                    "description": "Stochastic Gradient Descent",
                    "advantages": ["Simple", "Well understood", "Good for large datasets"],
                    "parameters": ["learning_rate", "momentum", "nesterovs_momentum"],
                    "use_case": "When you want fine control over optimization"
                },
                "lbfgs": {
                    "description": "Limited-memory Broyden-Fletcher-Goldfarb-Shanno",
                    "advantages": ["Fast convergence", "Good for small datasets"],
                    "disadvantages": ["Memory intensive", "Not suitable for large datasets"],
                    "use_case": "Small datasets (<1000 samples)"
                }
            },
            "strengths": [
                "Universal function approximation capability",
                "Can learn complex non-linear patterns",
                "Flexible architecture design",
                "Works well with large datasets",
                "Probabilistic outputs for classification",
                "Foundation of deep learning",
                "Automatic feature learning",
                "Good performance on many domains",
                "Handles high-dimensional data well",
                "Multiple optimization algorithms available",
                "Theoretical backing and understanding",
                "Scalable to large problems"
            ],
            "weaknesses": [
                "Requires careful hyperparameter tuning",
                "Sensitive to feature scaling (critical)",
                "Prone to overfitting on small datasets",
                "Black box - limited interpretability",
                "Local minima in optimization",
                "Computationally expensive training",
                "Sensitive to initialization",
                "May require many iterations to converge",
                "No guarantee of global optimum",
                "Requires large amounts of data for best performance",
                "Hyperparameter selection can be challenging",
                "Gradient vanishing/exploding problems"
            ],
            "ideal_use_cases": [
                "Complex pattern recognition tasks",
                "Image classification (with proper preprocessing)",
                "Natural language processing",
                "Large datasets with non-linear relationships",
                "Problems where interpretability is not critical",
                "Multi-class classification problems",
                "Function approximation tasks",
                "Time series prediction",
                "Recommendation systems",
                "Fraud detection",
                "Medical diagnosis",
                "Speech recognition",
                "Automated feature learning scenarios"
            ],
            "architecture_design_guide": {
                "number_of_layers": {
                    "1_hidden": "Linear separation with non-linear transformation",
                    "2_hidden": "Can represent any continuous function",
                    "3+_hidden": "Can learn hierarchical features but risk overfitting"
                },
                "layer_size_heuristics": [
                    "Start with (n_features + n_classes) / 2",
                    "Use powers of 2: 64, 128, 256, 512",
                    "Decrease size in later layers (pyramid shape)",
                    "For classification: final hidden layer ≥ n_classes"
                ],
                "common_architectures": {
                    "simple": "(100,) - single hidden layer with 100 neurons",
                    "moderate": "(100, 50) - two layers, decreasing size", 
                    "complex": "(200, 100, 50) - three layers, pyramid shape",
                    "wide": "(500,) - single wide layer",
                    "deep": "(64, 64, 64, 64) - deep uniform architecture"
                }
            },
            "hyperparameter_tuning_guide": {
                "architecture": "Start simple (100,), increase complexity if needed",
                "learning_rate": "0.001 for Adam, 0.01-0.1 for SGD", 
                "alpha": "Start with 0.0001, increase if overfitting",
                "max_iter": "200-1000 depending on dataset size",
                "early_stopping": "Use to prevent overfitting",
                "batch_size": "Auto (good default), or 32-512 for manual tuning"
            },
            "scaling_importance": {
                "critical_requirement": "Neural networks are extremely sensitive to input scale",
                "reason": "Weights are initialized with small random values",
                "consequence": "Large input values → large gradients → unstable training",
                "recommendation": "Always use StandardScaler or MinMaxScaler",
                "example": "Feature with range [0, 100000] will dominate features with range [0, 1]"
            },
            "overfitting_prevention": {
                "regularization": "Increase alpha parameter (L2 penalty)",
                "early_stopping": "Stop training when validation performance plateaus",
                "architecture": "Reduce number of hidden units or layers",
                "data": "Collect more training data",
                "dropout": "Not directly available in sklearn MLPClassifier"
            },
            "convergence_troubleshooting": {
                "slow_convergence": [
                    "Increase learning rate",
                    "Use better optimizer (Adam instead of SGD)",
                    "Check feature scaling",
                    "Increase max_iter"
                ],
                "no_convergence": [
                    "Decrease learning rate", 
                    "Increase tolerance",
                    "Check for data quality issues",
                    "Try different initialization (different random_state)"
                ],
                "unstable_training": [
                    "Decrease learning rate",
                    "Ensure proper feature scaling",
                    "Check for outliers in data",
                    "Use gradient clipping (not available in sklearn)"
                ]
            },
            "comparison_with_other_methods": {
                "vs_logistic_regression": {
                    "complexity": "MLP: high capacity, LR: linear only",
                    "interpretability": "MLP: black box, LR: transparent",
                    "data_requirements": "MLP: more data needed, LR: works with less",
                    "training_time": "MLP: longer, LR: very fast"
                },
                "vs_random_forest": {
                    "feature_scaling": "MLP: critical, RF: not needed",
                    "interpretability": "MLP: poor, RF: good feature importance",
                    "hyperparameter_tuning": "MLP: more complex, RF: more straightforward",
                    "overfitting": "MLP: more prone, RF: naturally resistant"
                },
                "vs_svm": {
                    "kernel_trick": "Both can handle non-linear patterns",
                    "probabilistic_output": "MLP: natural, SVM: needs calibration",
                    "training_time": "MLP: O(n*iter), SVM: O(n²) to O(n³)",
                    "parameter_sensitivity": "Both require careful tuning"
                }
            }
        }
    
    def fit(self, X, y, 
            monitor_training=True,
            compute_learning_curve=False):
        """
        Fit the MLP Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        monitor_training : bool, default=True
            Whether to monitor training progress
        compute_learning_curve : bool, default=False
            Whether to compute learning curve data
            
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
        
        # Feature scaling (CRITICAL for neural networks)
        if self.auto_scaling:
            if self.scaling_method == 'standard':
                self.scaler_ = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler_ = MinMaxScaler()
            else:
                self.scaler_ = StandardScaler()
            
            X_scaled = self.scaler_.fit_transform(X)
            
            # Check for scaling issues
            self._validate_scaling(X_scaled)
        else:
            X_scaled = X
            self.scaler_ = None
            warnings.warn("Neural networks require feature scaling! Consider enabling auto_scaling.")
        
        # Architecture optimization
        if self.architecture_optimization:
            optimal_architecture = self._optimize_architecture(X_scaled, y_encoded)
            hidden_layer_sizes = optimal_architecture
        else:
            hidden_layer_sizes = self.hidden_layer_sizes
        
        # Create and fit MLP model
        self.model_ = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
            warm_start=self.warm_start,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            n_iter_no_change=self.n_iter_no_change,
            max_fun=self.max_fun
        )
        
        # Train the model
        self.model_.fit(X_scaled, y_encoded)
        
        # Analyze training
        if monitor_training:
            self._analyze_training()
        
        # Compute learning curve
        if compute_learning_curve:
            self._compute_learning_curve(X_scaled, y_encoded)
        
        # Analyze architecture
        self._analyze_architecture()
        
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
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get predictions
        y_pred_encoded = self.model_.predict(X_scaled)
        
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
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get probability predictions
        probabilities = self.model_.predict_proba(X_scaled)
        
        return probabilities
    
    def _validate_scaling(self, X_scaled):
        """Validate that scaling was applied correctly"""
        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0)
        
        if self.scaling_method == 'standard':
            # Check if roughly standardized (mean≈0, std≈1)
            if np.any(np.abs(means) > 0.1) or np.any(np.abs(stds - 1) > 0.1):
                warnings.warn("Scaling may not have worked correctly. Check for constant features.")
        elif self.scaling_method == 'minmax':
            # Check if roughly in [0, 1] range
            mins = np.min(X_scaled, axis=0)
            maxs = np.max(X_scaled, axis=0)
            if np.any(mins < -0.1) or np.any(maxs > 1.1):
                warnings.warn("MinMax scaling may not have worked correctly.")
    
    def _optimize_architecture(self, X, y):
        """Optimize neural network architecture automatically"""
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        
        # Simple heuristic-based optimization
        architectures = [
            (n_features // 2,),
            (100,),
            (n_features, n_features // 2),
            (100, 50),
            (200, 100),
            (n_features * 2, n_features, n_features // 2)
        ]
        
        best_score = -np.inf
        best_architecture = self.hidden_layer_sizes
        
        for architecture in architectures:
            try:
                # Quick validation with small max_iter
                temp_mlp = MLPClassifier(
                    hidden_layer_sizes=architecture,
                    max_iter=50,
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.2
                )
                
                # Simple cross-validation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(temp_mlp, X, y, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_architecture = architecture
                    
            except Exception:
                continue
        
        return best_architecture
    
    def _analyze_training(self):
        """Analyze training process and convergence"""
        self.training_history_ = {
            "converged": getattr(self.model_, 'n_iter_', 0) < self.max_iter,
            "n_iterations": getattr(self.model_, 'n_iter_', 0),
            "max_iterations": self.max_iter,
            "final_loss": getattr(self.model_, 'loss_', None),
            "loss_curve": getattr(self.model_, 'loss_curve_', []),
            "validation_scores": getattr(self.model_, 'validation_scores_', [])
        }
        
        # Convergence analysis
        self.convergence_analysis_ = self._analyze_convergence()
    
    def _analyze_convergence(self):
        """Analyze convergence behavior"""
        if not self.training_history_ or not self.training_history_['loss_curve']:
            return {"status": "No convergence data available"}
        
        loss_curve = self.training_history_['loss_curve']
        
        analysis = {
            "converged": self.training_history_['converged'],
            "final_loss": loss_curve[-1] if loss_curve else None,
            "initial_loss": loss_curve[0] if loss_curve else None,
            "total_reduction": loss_curve[0] - loss_curve[-1] if len(loss_curve) > 0 else 0,
            "n_iterations": len(loss_curve),
            "convergence_rate": "fast" if len(loss_curve) < 50 else "moderate" if len(loss_curve) < 150 else "slow"
        }
        
        # Analyze loss curve stability
        if len(loss_curve) > 10:
            recent_losses = loss_curve[-10:]
            loss_std = np.std(recent_losses)
            analysis["stability"] = "stable" if loss_std < 0.01 else "unstable"
            analysis["recent_loss_std"] = loss_std
        
        # Check for overfitting signs
        validation_scores = self.training_history_.get('validation_scores', [])
        if validation_scores and len(validation_scores) > 10:
            # Simple overfitting detection
            recent_val_scores = validation_scores[-10:]
            if len(recent_val_scores) > 5:
                trend = np.polyfit(range(len(recent_val_scores)), recent_val_scores, 1)[0]
                analysis["overfitting_risk"] = "high" if trend < -0.001 else "low"
        
        return analysis
    
    def _compute_learning_curve(self, X, y):
        """Compute learning curve data"""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                self.model_, X, y, cv=3, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy'
            )
            
            self.learning_curve_data_ = {
                "train_sizes": train_sizes,
                "train_scores_mean": train_scores.mean(axis=1),
                "train_scores_std": train_scores.std(axis=1),
                "val_scores_mean": val_scores.mean(axis=1),
                "val_scores_std": val_scores.std(axis=1)
            }
        except Exception:
            self.learning_curve_data_ = None
    
    def _analyze_architecture(self):
        """Analyze the neural network architecture"""
        if not hasattr(self.model_, 'coefs_'):
            return
        
        # Get layer information
        layer_sizes = [self.model_.coefs_[0].shape[0]]  # Input layer
        for coef in self.model_.coefs_:
            layer_sizes.append(coef.shape[1])
        
        # Calculate total parameters
        total_params = sum(coef.size for coef in self.model_.coefs_)
        total_params += sum(intercept.size for intercept in self.model_.intercepts_)
        
        # Analyze weight distributions
        weight_stats = []
        for i, coef in enumerate(self.model_.coefs_):
            stats = {
                "layer": i + 1,
                "shape": coef.shape,
                "mean": float(np.mean(coef)),
                "std": float(np.std(coef)),
                "min": float(np.min(coef)),
                "max": float(np.max(coef))
            }
            weight_stats.append(stats)
        
        self.architecture_analysis_ = {
            "layer_sizes": layer_sizes,
            "total_parameters": total_params,
            "n_layers": len(layer_sizes),
            "hidden_layers": len(layer_sizes) - 2,
            "weight_statistics": weight_stats,
            "activation_function": self.activation,
            "solver_used": self.solver
        }
    
    def get_neural_network_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of neural network structure and training
        
        Returns:
        --------
        analysis_info : dict
            Comprehensive neural network analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "architecture_summary": {
                "input_size": self.n_features_in_,
                "hidden_layers": self.hidden_layer_sizes,
                "output_size": len(self.classes_),
                "total_parameters": self.architecture_analysis_['total_parameters'] if self.architecture_analysis_ else "Unknown",
                "activation_function": self.activation,
                "solver": self.solver
            },
            "training_configuration": {
                "max_iterations": self.max_iter,
                "learning_rate_init": self.learning_rate_init,
                "regularization_alpha": self.alpha,
                "batch_size": self.batch_size,
                "early_stopping": self.early_stopping,
                "feature_scaling": self.scaler_ is not None
            }
        }
        
        # Add training history
        if self.training_history_:
            analysis["training_results"] = {
                "converged": self.training_history_['converged'],
                "iterations_used": self.training_history_['n_iterations'],
                "final_loss": self.training_history_['final_loss'],
                "convergence_status": "Converged" if self.training_history_['converged'] else "Max iterations reached"
            }
        
        # Add convergence analysis
        if self.convergence_analysis_:
            analysis["convergence_analysis"] = self.convergence_analysis_
        
        # Add architecture details
        if self.architecture_analysis_:
            analysis["architecture_details"] = self.architecture_analysis_
        
        return analysis
    
    def plot_training_analysis(self, figsize=(15, 10)):
        """
        Create comprehensive training analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Training analysis visualization
        """
        if not self.is_fitted_ or not self.training_history_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Loss Curve
        loss_curve = self.training_history_['loss_curve']
        if loss_curve:
            iterations = range(1, len(loss_curve) + 1)
            ax1.plot(iterations, loss_curve, 'b-', linewidth=2, label='Training Loss')
            
            # Add validation scores if available
            val_scores = self.training_history_.get('validation_scores', [])
            if val_scores:
                val_iterations = range(1, len(val_scores) + 1)
                # Convert to loss (assuming accuracy scores)
                val_loss = [1 - score for score in val_scores]
                ax1.plot(val_iterations, val_loss, 'r--', linewidth=2, label='Validation Loss')
            
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('MLP Training Loss Curve')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_yscale('log')
        else:
            ax1.text(0.5, 0.5, 'No loss curve available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Training Loss Curve')
        
        # 2. Architecture Visualization
        if self.architecture_analysis_:
            layer_sizes = self.architecture_analysis_['layer_sizes']
            layer_names = ['Input'] + [f'Hidden {i}' for i in range(1, len(layer_sizes)-1)] + ['Output']
            
            ax2.bar(range(len(layer_sizes)), layer_sizes, color='skyblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Number of Neurons')
            ax2.set_title('Neural Network Architecture')
            ax2.set_xticks(range(len(layer_sizes)))
            ax2.set_xticklabels(layer_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, size in enumerate(layer_sizes):
                ax2.text(i, size + max(layer_sizes) * 0.01, str(size), 
                        ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Architecture analysis\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Network Architecture')
        
        # 3. Weight Distribution Analysis
        if self.architecture_analysis_ and 'weight_statistics' in self.architecture_analysis_:
            weight_stats = self.architecture_analysis_['weight_statistics']
            layers = [f"Layer {stat['layer']}" for stat in weight_stats]
            means = [stat['mean'] for stat in weight_stats]
            stds = [stat['std'] for stat in weight_stats]
            
            x = range(len(layers))
            ax3.bar([i - 0.2 for i in x], means, 0.4, label='Mean', alpha=0.7, color='green')
            ax3.bar([i + 0.2 for i in x], stds, 0.4, label='Std Dev', alpha=0.7, color='orange')
            
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Weight Value')
            ax3.set_title('Weight Distribution by Layer')
            ax3.set_xticks(x)
            ax3.set_xticklabels(layers, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Weight statistics\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Weight Distribution')
        
        # 4. Learning Curve (if available)
        if self.learning_curve_data_:
            lc_data = self.learning_curve_data_
            ax4.plot(lc_data['train_sizes'], lc_data['train_scores_mean'], 
                    'o-', color='blue', label='Training Score')
            ax4.fill_between(lc_data['train_sizes'], 
                           lc_data['train_scores_mean'] - lc_data['train_scores_std'],
                           lc_data['train_scores_mean'] + lc_data['train_scores_std'],
                           alpha=0.1, color='blue')
            
            ax4.plot(lc_data['train_sizes'], lc_data['val_scores_mean'], 
                    'o-', color='red', label='Validation Score')
            ax4.fill_between(lc_data['train_sizes'], 
                           lc_data['val_scores_mean'] - lc_data['val_scores_std'],
                           lc_data['val_scores_mean'] + lc_data['val_scores_std'],
                           alpha=0.1, color='red')
            
            ax4.set_xlabel('Training Set Size')
            ax4.set_ylabel('Accuracy Score')
            ax4.set_title('Learning Curves')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Show convergence information instead
            if self.convergence_analysis_:
                conv_info = self.convergence_analysis_
                info_text = f"""Convergence Analysis:
                
Status: {'Converged' if conv_info.get('converged', False) else 'Not Converged'}
Iterations: {conv_info.get('n_iterations', 'Unknown')}
Final Loss: {conv_info.get('final_loss', 'Unknown'):.6f}
Convergence Rate: {conv_info.get('convergence_rate', 'Unknown')}
Stability: {conv_info.get('stability', 'Unknown')}"""
                
                ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=9)
                ax4.set_title('Convergence Analysis')
                ax4.axis('off')
            else:
                ax4.text(0.5, 0.5, 'Learning curve\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Learning Curves')
        
        plt.tight_layout()
        return fig
    
    def plot_decision_boundary_2d(self, X, y, feature_indices=(0, 1), figsize=(10, 8), resolution=100):
        """
        Plot 2D decision boundary visualization (works only with 2D data or 2 selected features)
        
        Parameters:
        -----------
        X : array-like
            Feature data
        y : array-like
            Target data
        feature_indices : tuple, default=(0, 1)
            Indices of features to plot
        figsize : tuple, default=(10, 8)
            Figure size
        resolution : int, default=100
            Grid resolution for decision boundary
            
        Returns:
        --------
        fig : matplotlib figure
            Decision boundary plot
        """
        if not self.is_fitted_:
            return None
        
        if X.shape[1] < 2:
            return None
        
        # Select two features
        X_2d = X[:, feature_indices]
        
        # Apply scaling if used
        if self.scaler_ is not None:
            # Create a temporary scaler for 2D data
            temp_scaler = type(self.scaler_)()
            X_2d_scaled = temp_scaler.fit_transform(X_2d)
        else:
            X_2d_scaled = X_2d
        
        # Create a temporary 2D MLP model
        temp_mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        # Encode labels
        y_encoded = self.label_encoder_.transform(y)
        temp_mlp.fit(X_2d_scaled, y_encoded)
        
        # Create meshgrid
        x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
        y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Predict on meshgrid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = temp_mlp.predict_proba(mesh_points)[:, 1] if len(self.classes_) == 2 else temp_mlp.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot decision boundary
        if len(self.classes_) == 2:
            # Binary classification - show probability contours
            contour = ax.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap=plt.cm.RdYlBu)
            ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
            plt.colorbar(contour, ax=ax, label='Probability')
        else:
            # Multi-class classification - show class regions
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set3)
        
        # Plot data points
        scatter = ax.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], 
                           c=y_encoded, cmap=plt.cm.Set1, edgecolors='black', s=50)
        
        ax.set_xlabel(f'Feature {feature_indices[0]}' + 
                     (f' ({self.feature_names_[feature_indices[0]]})' if self.feature_names_ else ''))
        ax.set_ylabel(f'Feature {feature_indices[1]}' + 
                     (f' ({self.feature_names_[feature_indices[1]]})' if self.feature_names_ else ''))
        ax.set_title(f'MLP Decision Boundary\nArchitecture: {self.hidden_layer_sizes}')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🧠 Multi-layer Perceptron Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Architecture", "Training", "Optimization", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Network Architecture**")
            
            # Architecture type
            architecture_type = st.selectbox(
                "Architecture Type:",
                options=['Simple', 'Moderate', 'Complex', 'Custom'],
                index=0,
                help="Pre-defined architectures or custom design",
                key=f"{key_prefix}_architecture_type"
            )
            
            if architecture_type == 'Simple':
                hidden_layer_sizes = (100,)
                st.info("Single hidden layer with 100 neurons")
            elif architecture_type == 'Moderate':
                hidden_layer_sizes = (100, 50)
                st.info("Two hidden layers: 100 → 50 neurons")
            elif architecture_type == 'Complex':
                hidden_layer_sizes = (200, 100, 50)
                st.info("Three hidden layers: 200 → 100 → 50 neurons")
            else:  # Custom
                st.markdown("**Custom Architecture Design:**")
                n_hidden_layers = st.slider(
                    "Number of Hidden Layers:",
                    min_value=1,
                    max_value=5,
                    value=1,
                    help="More layers can learn complex patterns but may overfit",
                    key=f"{key_prefix}_n_hidden_layers"
                )
                
                layer_sizes = []
                for i in range(n_hidden_layers):
                    size = st.number_input(
                        f"Hidden Layer {i+1} Size:",
                        min_value=1,
                        max_value=1000,
                        value=100 if i == 0 else max(10, layer_sizes[i-1] // 2),
                        step=10,
                        help=f"Number of neurons in hidden layer {i+1}",
                        key=f"{key_prefix}_layer_{i}_size"
                    )
                    layer_sizes.append(size)
                
                hidden_layer_sizes = tuple(layer_sizes)
                st.info(f"Custom architecture: {' → '.join(map(str, hidden_layer_sizes))}")
            
            # Activation function
            activation = st.selectbox(
                "Activation Function:",
                options=['relu', 'tanh', 'logistic', 'identity'],
                index=0,
                help="Non-linear activation function",
                key=f"{key_prefix}_activation"
            )
            
            # Architecture optimization
            architecture_optimization = st.checkbox(
                "Auto-Optimize Architecture",
                value=self.architecture_optimization,
                help="Automatically find the best architecture using cross-validation",
                key=f"{key_prefix}_architecture_optimization"
            )
        
        with tab2:
            st.markdown("**Training Configuration**")
            
            # Max iterations
            max_iter = st.slider(
                "Max Iterations:",
                min_value=50,
                max_value=1000,
                value=int(self.max_iter),
                step=50,
                help="Maximum number of training iterations",
                key=f"{key_prefix}_max_iter"
            )
            
            # Early stopping
            early_stopping = st.checkbox(
                "Early Stopping",
                value=self.early_stopping,
                help="Stop training when validation score stops improving",
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
                    "Patience:",
                    min_value=5,
                    max_value=50,
                    value=int(self.n_iter_no_change),
                    help="Iterations with no improvement before stopping",
                    key=f"{key_prefix}_n_iter_no_change"
                )
            else:
                validation_fraction = self.validation_fraction
                n_iter_no_change = self.n_iter_no_change
            
            # Tolerance
            tol = st.number_input(
                "Tolerance:",
                value=float(self.tol),
                min_value=1e-6,
                max_value=1e-2,
                step=1e-4,
                format="%.6f",
                help="Tolerance for optimization",
                key=f"{key_prefix}_tol"
            )
            
            # Shuffle
            shuffle = st.checkbox(
                "Shuffle Data",
                value=self.shuffle,
                help="Shuffle samples in each iteration",
                key=f"{key_prefix}_shuffle"
            )
        
        with tab3:
            st.markdown("**Optimization Settings**")
            
            # Solver
            solver = st.selectbox(
                "Optimizer:",
                options=['adam', 'sgd', 'lbfgs'],
                index=['adam', 'sgd', 'lbfgs'].index(self.solver),
                help="Optimization algorithm",
                key=f"{key_prefix}_solver"
            )
            
            # Learning rate
            learning_rate_init = st.number_input(
                "Learning Rate:",
                value=float(self.learning_rate_init),
                min_value=1e-5,
                max_value=1.0,
                step=1e-4,
                format="%.5f",
                help="Initial learning rate",
                key=f"{key_prefix}_learning_rate_init"
            )
            
            # Learning rate schedule
            learning_rate = st.selectbox(
                "Learning Rate Schedule:",
                options=['constant', 'invscaling', 'adaptive'],
                index=['constant', 'invscaling', 'adaptive'].index(self.learning_rate),
                help="Learning rate schedule for weight updates",
                key=f"{key_prefix}_learning_rate"
            )
            
            # Regularization
            alpha = st.number_input(
                "Regularization (Alpha):",
                value=float(self.alpha),
                min_value=1e-6,
                max_value=1.0,
                step=1e-4,
                format="%.6f",
                help="L2 penalty parameter",
                key=f"{key_prefix}_alpha"
            )
            
            # Batch size
            batch_size_option = st.selectbox(
                "Batch Size:",
                options=['auto', '32', '64', '128', '256', '512'],
                index=0,
                help="Size of minibatches for stochastic optimizers",
                key=f"{key_prefix}_batch_size_option"
            )
            
            batch_size = 'auto' if batch_size_option == 'auto' else int(batch_size_option)
            
            # Solver-specific parameters
            if solver == 'adam':
                st.markdown("**Adam Parameters:**")
                beta_1 = st.slider(
                    "Beta 1:",
                    min_value=0.8,
                    max_value=0.99,
                    value=float(self.beta_1),
                    step=0.01,
                    help="Exponential decay rate for first moment estimates",
                    key=f"{key_prefix}_beta_1"
                )
                
                beta_2 = st.slider(
                    "Beta 2:",
                    min_value=0.9,
                    max_value=0.9999,
                    value=float(self.beta_2),
                    step=0.0001,
                    help="Exponential decay rate for second moment estimates",
                    key=f"{key_prefix}_beta_2"
                )
                
                epsilon = st.number_input(
                    "Epsilon:",
                    value=float(self.epsilon),
                    min_value=1e-10,
                    max_value=1e-6,
                    step=1e-8,
                    format="%.2e",
                    help="Value for numerical stability",
                    key=f"{key_prefix}_epsilon"
                )
            else:
                beta_1 = self.beta_1
                beta_2 = self.beta_2
                epsilon = self.epsilon
            
            if solver == 'sgd':
                st.markdown("**SGD Parameters:**")
                momentum = st.slider(
                    "Momentum:",
                    min_value=0.0,
                    max_value=0.99,
                    value=float(self.momentum),
                    step=0.01,
                    help="Momentum for gradient descent update",
                    key=f"{key_prefix}_momentum"
                )
                
                nesterovs_momentum = st.checkbox(
                    "Nesterov's Momentum",
                    value=self.nesterovs_momentum,
                    help="Whether to use Nesterov's momentum",
                    key=f"{key_prefix}_nesterovs_momentum"
                )
            else:
                momentum = self.momentum
                nesterovs_momentum = self.nesterovs_momentum
        
        with tab4:
            st.markdown("**Advanced Settings**")
            
            # Feature scaling
            auto_scaling = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scaling,
                help="Automatically scale features (CRITICAL for neural networks)",
                key=f"{key_prefix}_auto_scaling"
            )
            
            if auto_scaling:
                scaling_method = st.selectbox(
                    "Scaling Method:",
                    options=['standard', 'minmax'],
                    index=['standard', 'minmax'].index(self.scaling_method),
                    help="standard: mean=0, std=1; minmax: range [0,1]",
                    key=f"{key_prefix}_scaling_method"
                )
            else:
                scaling_method = self.scaling_method
                st.error("⚠️ Neural networks require feature scaling!")
            
            # Random state
            random_state = st.number_input(
                "Random Seed:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
            
            # Warm start
            warm_start = st.checkbox(
                "Warm Start",
                value=self.warm_start,
                help="Reuse solution of previous call to fit",
                key=f"{key_prefix}_warm_start"
            )
            
            # Verbose
            verbose = st.checkbox(
                "Verbose Output",
                value=self.verbose,
                help="Print progress messages during training",
                key=f"{key_prefix}_verbose"
            )
            
            # Analysis options
            st.markdown("**Analysis Options:**")
            compute_learning_curve = st.checkbox(
                "Compute Learning Curve",
                value=False,
                help="Compute learning curve (may slow down training)",
                key=f"{key_prefix}_compute_learning_curve"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if SKLEARN_AVAILABLE:
                st.success("✅ scikit-learn is available")
            else:
                st.error("❌ scikit-learn not installed. Run: pip install scikit-learn")
            
            st.info("""
            **Multi-layer Perceptron** - Standard Neural Network:
            • 🧠 Universal function approximator
            • 🔄 Feedforward architecture with backpropagation
            • 📈 Multiple hidden layers for complex patterns
            • 🎯 Gradient-based optimization
            • 📊 Probabilistic classification outputs
            • 🌐 Foundation of deep learning
            
            **Key Characteristics:**
            • Non-linear decision boundaries
            • Automatic feature learning
            • Requires feature scaling (critical!)
            • Iterative gradient-based training
            """)
            
            # Architecture guide
            if st.button("🏗️ Architecture Design Guide", key=f"{key_prefix}_architecture_guide"):
                st.markdown("""
                **Neural Network Architecture Design:**
                
                **Number of Hidden Layers:**
                - 1 layer: Can approximate any continuous function
                - 2-3 layers: Good for most problems
                - 4+ layers: Risk of overfitting, need more data
                
                **Layer Size Guidelines:**
                - Start with (n_features + n_classes) / 2
                - Common sizes: 50, 100, 200, 500
                - Pyramid shape: decreasing size in later layers
                
                **Common Architectures:**
                - Simple: (100,) - good starting point
                - Moderate: (100, 50) - for complex problems
                - Deep: (128, 64, 32) - for very complex data
                """)
            
            # Activation functions guide
            if st.button("⚡ Activation Functions", key=f"{key_prefix}_activation_guide"):
                st.markdown("""
                **Choosing Activation Functions:**
                
                **ReLU (Recommended):**
                - f(x) = max(0, x)
                - Fast computation, avoids vanishing gradient
                - Best choice for hidden layers
                
                **Tanh:**
                - f(x) = tanh(x), output in (-1, 1)
                - Zero-centered, good gradient flow
                - Alternative to ReLU
                
                **Logistic (Sigmoid):**
                - f(x) = 1/(1+e^(-x)), output in (0, 1)
                - Historically important, now less used
                - Can cause vanishing gradients
                
                **Identity (Linear):**
                - f(x) = x
                - Only for output layer in regression
                """)
            
            # Optimizer guide
            if st.button("🎯 Optimizer Selection", key=f"{key_prefix}_optimizer_guide"):
                st.markdown("""
                **Optimization Algorithm Guide:**
                
                **Adam (Recommended):**
                - Adaptive learning rates per parameter
                - Works well in most cases
                - Good default choice
                
                **SGD with Momentum:**
                - Classic approach, well understood
                - Good for fine-tuning
                - Requires more careful learning rate tuning
                
                **L-BFGS:**
                - Fast convergence for small datasets
                - Memory intensive
                - Good for <1000 samples
                
                **Learning Rate Guidelines:**
                - Adam: 0.001 (default)
                - SGD: 0.01-0.1
                - Too high: unstable training
                - Too low: slow convergence
                """)
            
            # Feature scaling warning
            if st.button("⚠️ Feature Scaling Critical!", key=f"{key_prefix}_scaling_warning"):
                st.markdown("""
                **Why Feature Scaling is CRITICAL for Neural Networks:**
                
                **Problem:**
                - Neural networks use random weight initialization
                - Features with large scales dominate the gradients
                - Different feature scales → unstable training
                
                **Example:**
                - Age: 0-100 vs Income: 0-100,000
                - Income will completely dominate the learning
                
                **Solution:**
                - StandardScaler: mean=0, std=1
                - MinMaxScaler: range [0,1]
                
                **Always scale features before training!**
                """)
            
            # Troubleshooting guide
            if st.button("🔧 Troubleshooting Guide", key=f"{key_prefix}_troubleshooting"):
                st.markdown("""
                **Common Issues and Solutions:**
                
                **Not Converging:**
                - Decrease learning rate
# Continue from line 1489 where the code breaks:
                - Increase max_iter
                - Check feature scaling
                - Try different random_state
                
                **Overfitting:**
                - Increase alpha (regularization)
                - Enable early_stopping
                - Reduce network complexity
                - Get more training data
                
                **Poor Performance:**
                - Check feature scaling (most common issue)
                - Try different architecture
                - Adjust learning rate
                - Increase training iterations
                
                **Unstable Training:**
                - Decrease learning rate
                - Ensure feature scaling
                - Check for outliers
                - Try different solver
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_init": learning_rate_init,
            "max_iter": max_iter,
            "shuffle": shuffle,
            "random_state": random_state,
            "tol": tol,
            "early_stopping": early_stopping,
            "validation_fraction": validation_fraction,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "epsilon": epsilon,
            "n_iter_no_change": n_iter_no_change,
            "momentum": momentum,
            "nesterovs_momentum": nesterovs_momentum,
            "auto_scaling": auto_scaling,
            "scaling_method": scaling_method,
            "architecture_optimization": architecture_optimization,
            "warm_start": warm_start,
            "verbose": verbose,
            "_ui_options": {
                "compute_learning_curve": compute_learning_curve,
                "show_training_analysis": True,
                "show_architecture_viz": True,
                "show_decision_boundary": True
            }
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return MLPClassifierPlugin(
            hidden_layer_sizes=hyperparameters.get("hidden_layer_sizes", self.hidden_layer_sizes),
            activation=hyperparameters.get("activation", self.activation),
            solver=hyperparameters.get("solver", self.solver),
            alpha=hyperparameters.get("alpha", self.alpha),
            batch_size=hyperparameters.get("batch_size", self.batch_size),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            learning_rate_init=hyperparameters.get("learning_rate_init", self.learning_rate_init),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            shuffle=hyperparameters.get("shuffle", self.shuffle),
            random_state=hyperparameters.get("random_state", self.random_state),
            tol=hyperparameters.get("tol", self.tol),
            early_stopping=hyperparameters.get("early_stopping", self.early_stopping),
            validation_fraction=hyperparameters.get("validation_fraction", self.validation_fraction),
            beta_1=hyperparameters.get("beta_1", self.beta_1),
            beta_2=hyperparameters.get("beta_2", self.beta_2),
            epsilon=hyperparameters.get("epsilon", self.epsilon),
            n_iter_no_change=hyperparameters.get("n_iter_no_change", self.n_iter_no_change),
            momentum=hyperparameters.get("momentum", self.momentum),
            nesterovs_momentum=hyperparameters.get("nesterovs_momentum", self.nesterovs_momentum),
            auto_scaling=hyperparameters.get("auto_scaling", self.auto_scaling),
            scaling_method=hyperparameters.get("scaling_method", self.scaling_method),
            architecture_optimization=hyperparameters.get("architecture_optimization", self.architecture_optimization),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            verbose=hyperparameters.get("verbose", self.verbose)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for MLP
        
        MLP requires feature scaling and doesn't handle missing values well.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # Check for missing values
        if np.any(pd.isna(X_processed)):
            warnings.warn("MLP doesn't handle missing values. Consider imputation before training.")
        
        # Check for infinite values
        if np.any(np.isinf(X_processed)):
            warnings.warn("MLP doesn't handle infinite values. Please clean your data.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if MLP is compatible with the given data
        
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
            return False, f"MLP requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for missing values
        if np.any(pd.isna(X)):
            return False, "MLP doesn't handle missing values well. Please impute missing values first."
        
        # Check for infinite values
        if np.any(np.isinf(X)):
            return False, "MLP doesn't handle infinite values. Please clean your data first."
        
        # Check for very high dimensionality
        if X.shape[1] > 1000:
            return True, "Warning: Very high dimensionality. Consider dimensionality reduction for better performance."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            # Check for class imbalance
            class_counts = np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes)
            min_class_size = np.min(class_counts)
            if min_class_size < 10:
                return True, f"Warning: Small class detected ({min_class_size} samples). Consider class balancing."
        
        return True, "MLP is compatible with this data. Remember to scale features!"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "convergence_status": self.training_history_['converged'] if self.training_history_ else None,
            "n_iterations_used": self.training_history_['n_iterations'] if self.training_history_ else None,
            "final_loss": self.training_history_['final_loss'] if self.training_history_ else None,
            "total_parameters": self.architecture_analysis_['total_parameters'] if self.architecture_analysis_ else None,
            "scaling_applied": self.scaler_ is not None,
            "scaling_method": self.scaling_method if self.scaler_ is not None else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Multi-layer Perceptron",
            "training_completed": True,
            "neural_network_characteristics": {
                "feedforward_architecture": True,
                "backpropagation_training": True,
                "gradient_based_optimization": True,
                "universal_approximator": True,
                "non_linear_activation": True,
                "iterative_learning": True
            },
            "model_configuration": {
                "hidden_layers": len(self.hidden_layer_sizes),
                "layer_sizes": list(self.hidden_layer_sizes),
                "activation_function": self.activation,
                "optimizer": self.solver,
                "regularization": self.alpha,
                "feature_scaling": self.scaler_ is not None
            },
            "neural_network_analysis": self.get_neural_network_analysis(),
            "performance_considerations": {
                "memory_usage": "Moderate - stores network weights",
                "prediction_time": "Fast - matrix multiplication",
                "training_time": "Moderate to slow - iterative optimization",
                "scalability": "Good for medium to large datasets",
                "hyperparameter_sensitivity": "High - requires careful tuning",
                "feature_scaling_dependency": "Critical - must scale features"
            },
            "optimization_insights": {
                "gradient_based": "Uses backpropagation for gradient computation",
                "local_minima": "May get stuck in local optima",
                "initialization_sensitivity": "Sensitive to weight initialization",
                "learning_rate_importance": "Critical parameter for convergence"
            }
        }
        
        return info


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return MLPClassifierPlugin()

