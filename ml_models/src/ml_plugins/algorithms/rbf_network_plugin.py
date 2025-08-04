import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.linalg import pinv

# Try to import required libraries with graceful fallback
try:
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import accuracy_score, classification_report
    import scipy.stats as stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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

class RBFNetworkPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Radial Basis Function (RBF) Network Plugin - Advanced Neural Network
    
    RBF networks are a type of artificial neural network that uses radial basis functions
    as activation functions. They consist of three layers: input layer, hidden layer with
    RBF neurons, and output layer. RBF networks are universal approximators and excel
    at function approximation and pattern recognition tasks.
    
    The network structure:
    1. Input Layer: Receives input features
    2. Hidden Layer: RBF neurons with Gaussian (or other) basis functions
    3. Output Layer: Linear combination of hidden layer outputs
    """
    
    def __init__(self,
                 n_centers=10,
                 rbf_function='gaussian',
                 center_selection='kmeans',
                 gamma='auto',
                 regularization=0.0,
                 normalize_inputs=True,
                 normalize_outputs=False,
                 random_state=42,
                 kmeans_init='k-means++',
                 max_iter=300,
                 tol=1e-4,
                 # Advanced parameters
                 adaptive_centers=False,
                 center_optimization='gradient',
                 width_optimization='heuristic',
                 output_layer_solver='analytical',
                 early_stopping=False,
                 validation_fraction=0.1,
                 learning_rate=0.01,
                 momentum=0.9,
                 # Custom parameters
                 auto_scaling=True,
                 scaling_method='standard',
                 verbose=False):
        """
        Initialize RBF Network with comprehensive parameter support
        
        Parameters:
        -----------
        n_centers : int, default=10
            Number of RBF centers (hidden neurons)
        rbf_function : str, default='gaussian'
            Type of radial basis function ('gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline')
        center_selection : str, default='kmeans'
            Method for selecting RBF centers ('kmeans', 'random', 'grid', 'data_points')
        gamma : float or 'auto', default='auto'
            RBF kernel parameter (width/spread of basis functions)
        regularization : float, default=0.0
            L2 regularization parameter for output weights
        normalize_inputs : bool, default=True
            Whether to normalize input data
        normalize_outputs : bool, default=False
            Whether to normalize output data
        random_state : int, default=42
            Random seed for reproducibility
        kmeans_init : str, default='k-means++'
            K-means initialization method
        max_iter : int, default=300
            Maximum iterations for training
        tol : float, default=1e-4
            Tolerance for convergence
        adaptive_centers : bool, default=False
            Whether to adapt center positions during training
        center_optimization : str, default='gradient'
            Method for center optimization ('gradient', 'genetic', 'none')
        width_optimization : str, default='heuristic'
            Method for width optimization ('heuristic', 'adaptive', 'fixed')
        output_layer_solver : str, default='analytical'
            Solver for output layer weights ('analytical', 'gradient', 'lbfgs')
        early_stopping : bool, default=False
            Whether to use early stopping
        validation_fraction : float, default=0.1
            Fraction of data for validation
        learning_rate : float, default=0.01
            Learning rate for gradient-based optimization
        momentum : float, default=0.9
            Momentum for gradient descent
        auto_scaling : bool, default=True
            Whether to automatically scale features
        scaling_method : str, default='standard'
            Scaling method ('standard', 'minmax')
        verbose : bool, default=False
            Whether to print training progress
        """
        super().__init__()
        
        # Core RBF parameters
        self.n_centers = n_centers
        self.rbf_function = rbf_function
        self.center_selection = center_selection
        self.gamma = gamma
        self.regularization = regularization
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.random_state = random_state
        self.kmeans_init = kmeans_init
        self.max_iter = max_iter
        self.tol = tol
        
        # Advanced parameters
        self.adaptive_centers = adaptive_centers
        self.center_optimization = center_optimization
        self.width_optimization = width_optimization
        self.output_layer_solver = output_layer_solver
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Custom parameters
        self.auto_scaling = auto_scaling
        self.scaling_method = scaling_method
        self.verbose = verbose
        
        # Plugin metadata
        self._name = "RBF Network"
        self._description = "Radial Basis Function network for complex pattern recognition and function approximation."
        self._category = "Neural Networks"
        self._algorithm_type = "Radial Basis Function Network"
        self._paper_reference = "Broomhead, D. S., & Lowe, D. (1988). Radial basis functions, multi-variable functional interpolation and adaptive networks."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False  # Can be extended for regression
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 20
        self._handles_missing_values = False
        self._requires_scaling = True
        self._supports_sparse = False
        self._is_linear = False
        self._provides_feature_importance = False
        self._provides_probabilities = True
        self._handles_categorical = False
        self._ensemble_method = False
        self._universal_approximator = True
        self._local_approximation = True
        self._center_based = True
        self._gaussian_basis = True
        self._three_layer_architecture = True
        
        # Internal attributes
        self.centers_ = None
        self.widths_ = None
        self.output_weights_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.training_history_ = None
        self.rbf_analysis_ = None
        self.center_analysis_ = None
        
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
            "year_introduced": 1988,
            "key_innovations": {
                "radial_basis_functions": "Uses distance-based activation functions",
                "local_approximation": "Each RBF neuron responds to local regions of input space",
                "universal_approximation": "Can approximate any continuous function",
                "three_layer_architecture": "Input → RBF Hidden Layer → Linear Output",
                "center_based_learning": "Learning involves positioning centers and adjusting weights",
                "gaussian_kernels": "Typically uses Gaussian functions as basis functions"
            },
            "algorithm_mechanics": {
                "network_structure": {
                    "input_layer": "Receives input features (no processing)",
                    "hidden_layer": "RBF neurons with radial basis functions",
                    "output_layer": "Linear combination of hidden layer outputs",
                    "activation": "Gaussian or other radial basis functions"
                },
                "rbf_computation": {
                    "step_1": "Calculate distance from input to each center",
                    "step_2": "Apply radial basis function (e.g., Gaussian)",
                    "step_3": "Weight and sum RBF outputs",
                    "step_4": "Apply output activation (linear for classification)"
                },
                "training_phases": {
                    "phase_1": "Position RBF centers (unsupervised)",
                    "phase_2": "Determine RBF widths",
                    "phase_3": "Train output layer weights (supervised)"
                },
                "gaussian_rbf": {
                    "formula": "φ(||x - c||) = exp(-γ||x - c||²)",
                    "parameters": "c = center, γ = width parameter",
                    "properties": "Smooth, differentiable, local response"
                }
            },
            "rbf_functions": {
                "gaussian": {
                    "formula": "exp(-γ||x - c||²)",
                    "properties": ["Smooth", "Infinitely differentiable", "Most popular"],
                    "advantages": ["Good generalization", "Stable training"],
                    "use_case": "Default choice for most problems"
                },
                "multiquadric": {
                    "formula": "sqrt(||x - c||² + γ²)",
                    "properties": ["Globally supported", "Non-compact"],
                    "advantages": ["Good for scattered data"],
                    "use_case": "When global influence is desired"
                },
                "inverse_multiquadric": {
                    "formula": "1/sqrt(||x - c||² + γ²)",
                    "properties": ["Globally supported", "Bounded"],
                    "advantages": ["Numerically stable"],
                    "use_case": "Alternative to Gaussian"
                },
                "thin_plate_spline": {
                    "formula": "||x - c||² * log(||x - c||)",
                    "properties": ["Polynomial growth", "Smooth"],
                    "advantages": ["Good interpolation properties"],
                    "use_case": "Function approximation tasks"
                }
            },
            "center_selection_methods": {
                "kmeans": {
                    "description": "Use K-means clustering to find centers",
                    "advantages": ["Data-driven", "Well-distributed centers"],
                    "disadvantages": ["Computationally expensive"],
                    "use_case": "Default choice for most problems"
                },
                "random": {
                    "description": "Randomly select data points as centers",
                    "advantages": ["Fast", "Simple"],
                    "disadvantages": ["May not be optimal"],
                    "use_case": "Quick prototyping or large datasets"
                },
                "grid": {
                    "description": "Place centers on a regular grid",
                    "advantages": ["Uniform coverage", "Predictable"],
                    "disadvantages": ["Curse of dimensionality"],
                    "use_case": "Low-dimensional problems"
                },
                "data_points": {
                    "description": "Use all or subset of training data as centers",
                    "advantages": ["Guaranteed coverage", "Simple"],
                    "disadvantages": ["Many centers needed"],
                    "use_case": "Small datasets"
                }
            },
            "strengths": [
                "Universal function approximation capability",
                "Fast training (analytical solution for output weights)",
                "Good interpolation properties",
                "Local learning (changes affect local regions)",
                "Interpretable structure (centers have meaning)",
                "Works well with small to medium datasets",
                "No vanishing gradient problem",
                "Stable and well-understood mathematically",
                "Flexible basis function selection",
                "Good for non-linear pattern recognition",
                "Effective for function approximation",
                "Less prone to overfitting than MLPs"
            ],
            "weaknesses": [
                "Curse of dimensionality (exponential growth of centers needed)",
                "Sensitive to center placement",
                "May require many centers for complex functions",
                "Width parameter selection is critical",
                "Not as popular as modern deep learning methods",
                "Can be computationally expensive for large datasets",
                "Limited scalability to very high dimensions",
                "Performance depends heavily on hyperparameter tuning",
                "May struggle with very sparse data",
                "Less effective than modern methods for very large datasets"
            ],
            "ideal_use_cases": [
                "Function approximation tasks",
                "Pattern recognition with medium-sized datasets",
                "Problems requiring fast training",
                "Applications needing interpretable models",
                "Time series prediction",
                "Control systems and signal processing",
                "Medical diagnosis and bioinformatics",
                "Image recognition (with appropriate preprocessing)",
                "Regression problems with smooth functions",
                "Classification with well-separated classes",
                "Scientific computing applications",
                "Real-time applications requiring fast prediction"
            ],
            "comparison_with_mlp": {
                "training_speed": "RBF: Faster (analytical), MLP: Slower (iterative)",
                "architecture": "RBF: Fixed 3-layer, MLP: Flexible multi-layer",
                "activation": "RBF: Radial basis functions, MLP: Sigmoid/ReLU",
                "learning": "RBF: Hybrid (unsupervised + supervised), MLP: Supervised",
                "locality": "RBF: Local response, MLP: Global response",
                "interpolation": "RBF: Excellent, MLP: Good",
                "scalability": "RBF: Limited, MLP: Better"
            },
            "mathematical_foundation": {
                "basis_function_theory": "Grounded in functional analysis and approximation theory",
                "universal_approximation": "Proven capability to approximate continuous functions",
                "regularization_theory": "Well-established mathematical framework",
                "interpolation_theory": "Strong connections to scattered data interpolation"
            },
            "hyperparameter_guide": {
                "n_centers": {
                    "rule_of_thumb": "Start with sqrt(n_samples), adjust based on complexity",
                    "range": "5-50 for most problems",
                    "considerations": "More centers = more complexity, risk of overfitting"
                },
                "gamma": {
                    "auto_calculation": "Based on average distance between centers",
                    "manual_tuning": "Start with 1.0, adjust based on data spread",
                    "effect": "Larger γ = narrower basis functions = more local"
                },
                "regularization": {
                    "purpose": "Prevent overfitting in output layer",
                    "range": "0.0 to 0.1 typically",
                    "when_to_use": "When you have many centers relative to data"
                }
            }
        }
    
    def _rbf_gaussian(self, distances, gamma):
        """Gaussian RBF function"""
        return np.exp(-gamma * distances**2)
    
    def _rbf_multiquadric(self, distances, gamma):
        """Multiquadric RBF function"""
        return np.sqrt(distances**2 + gamma**2)
    
    def _rbf_inverse_multiquadric(self, distances, gamma):
        """Inverse multiquadric RBF function"""
        return 1.0 / np.sqrt(distances**2 + gamma**2)
    
    def _rbf_thin_plate_spline(self, distances, gamma):
        """Thin plate spline RBF function"""
        # Avoid log(0) by adding small epsilon
        eps = 1e-12
        return np.where(distances > eps, distances**2 * np.log(distances + eps), 0.0)
    
    def _apply_rbf_function(self, distances):
        """Apply the selected RBF function"""
        if self.rbf_function == 'gaussian':
            return self._rbf_gaussian(distances, self.gamma_)
        elif self.rbf_function == 'multiquadric':
            return self._rbf_multiquadric(distances, self.gamma_)
        elif self.rbf_function == 'inverse_multiquadric':
            return self._rbf_inverse_multiquadric(distances, self.gamma_)
        elif self.rbf_function == 'thin_plate_spline':
            return self._rbf_thin_plate_spline(distances, self.gamma_)
        else:
            raise ValueError(f"Unknown RBF function: {self.rbf_function}")
    
    def _select_centers(self, X):
        """Select RBF centers using the specified method"""
        n_samples, n_features = X.shape
        
        if self.center_selection == 'kmeans':
            if self.n_centers > n_samples:
                warnings.warn(f"n_centers ({self.n_centers}) > n_samples ({n_samples}). Using all data points as centers.")
                return X.copy()
            
            kmeans = KMeans(
                n_clusters=self.n_centers,
                init=self.kmeans_init,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            kmeans.fit(X)
            return kmeans.cluster_centers_
            
        elif self.center_selection == 'random':
            np.random.seed(self.random_state)
            n_centers_actual = min(self.n_centers, n_samples)
            indices = np.random.choice(n_samples, n_centers_actual, replace=False)
            return X[indices]
            
        elif self.center_selection == 'data_points':
            if self.n_centers >= n_samples:
                return X.copy()
            np.random.seed(self.random_state)
            indices = np.random.choice(n_samples, self.n_centers, replace=False)
            return X[indices]
            
        elif self.center_selection == 'grid':
            if n_features > 3:
                warnings.warn("Grid selection not recommended for high dimensions. Falling back to kmeans.")
                return self._select_centers_kmeans(X)
            
            # Simple grid for low dimensions
            n_per_dim = max(2, int(self.n_centers**(1/n_features)))
            ranges = []
            for i in range(n_features):
                min_val, max_val = X[:, i].min(), X[:, i].max()
                ranges.append(np.linspace(min_val, max_val, n_per_dim))
            
            centers = np.array(np.meshgrid(*ranges)).T.reshape(-1, n_features)
            return centers[:self.n_centers]
            
        else:
            raise ValueError(f"Unknown center selection method: {self.center_selection}")
    
    def _determine_widths(self, centers):
        """Determine RBF widths based on center distances"""
        if self.width_optimization == 'heuristic':
            # Use average distance to nearest neighbor
            if len(centers) > 1:
                distances = euclidean_distances(centers, centers)
                # Set diagonal to infinity to exclude self-distances
                np.fill_diagonal(distances, np.inf)
                nearest_distances = np.min(distances, axis=1)
                avg_distance = np.mean(nearest_distances)
                # Width parameter (gamma) is inverse related to distance
                self.gamma_ = 1.0 / (2 * avg_distance**2) if avg_distance > 0 else 1.0
            else:
                self.gamma_ = 1.0
        elif self.width_optimization == 'fixed':
            self.gamma_ = 1.0 if self.gamma == 'auto' else self.gamma
        elif self.width_optimization == 'adaptive':
            # Individual widths for each center (not implemented in this version)
            self.gamma_ = 1.0 if self.gamma == 'auto' else self.gamma
        else:
            self.gamma_ = 1.0 if self.gamma == 'auto' else self.gamma
    
    def _compute_rbf_matrix(self, X, centers):
        """Compute the RBF activation matrix"""
        # Calculate distances from each data point to each center
        distances = euclidean_distances(X, centers)
        
        # Apply RBF function
        rbf_matrix = self._apply_rbf_function(distances)
        
        return rbf_matrix
    
    def _solve_output_weights(self, rbf_matrix, y_encoded):
        """Solve for output layer weights"""
        n_classes = len(self.classes_)
        
        # Convert labels to one-hot encoding for multi-class
        if n_classes > 2:
            y_onehot = np.zeros((len(y_encoded), n_classes))
            y_onehot[np.arange(len(y_encoded)), y_encoded] = 1
        else:
            y_onehot = y_encoded.reshape(-1, 1)
        
        if self.output_layer_solver == 'analytical':
            # Analytical solution using pseudoinverse with regularization
            if self.regularization > 0:
                # Ridge regression solution
                A = rbf_matrix.T @ rbf_matrix + self.regularization * np.eye(rbf_matrix.shape[1])
                b = rbf_matrix.T @ y_onehot
                self.output_weights_ = np.linalg.solve(A, b)
            else:
                # Pseudoinverse solution
                self.output_weights_ = pinv(rbf_matrix) @ y_onehot
                
        elif self.output_layer_solver == 'gradient':
            # Gradient descent solution (simplified implementation)
            n_centers = rbf_matrix.shape[1]
            self.output_weights_ = np.random.randn(n_centers, n_classes if n_classes > 2 else 1) * 0.1
            
            for iteration in range(self.max_iter):
                # Forward pass
                predictions = rbf_matrix @ self.output_weights_
                
                # Compute error
                error = predictions - y_onehot
                
                # Compute gradient
                gradient = rbf_matrix.T @ error / len(y_onehot)
                
                # Update weights
                self.output_weights_ -= self.learning_rate * gradient
                
                # Check convergence
                if np.linalg.norm(gradient) < self.tol:
                    break
        else:
            raise ValueError(f"Unknown output layer solver: {self.output_layer_solver}")
    
    def fit(self, X, y, 
            monitor_training=True,
            compute_learning_curve=False):
        """
        Fit the RBF Network model
        
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
        
        # Feature scaling (important for RBF networks)
        if self.auto_scaling:
            if self.scaling_method == 'standard':
                self.scaler_ = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler_ = MinMaxScaler()
            else:
                self.scaler_ = StandardScaler()
            
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
            warnings.warn("RBF networks benefit from feature scaling!")
        
        if self.verbose:
            print(f"Training RBF Network with {self.n_centers} centers...")
        
        # Phase 1: Select RBF centers (unsupervised)
        self.centers_ = self._select_centers(X_scaled)
        
        if self.verbose:
            print(f"Selected {len(self.centers_)} centers using {self.center_selection} method")
        
        # Phase 2: Determine RBF widths
        self._determine_widths(self.centers_)
        
        if self.verbose:
            print(f"Determined RBF width parameter: γ = {self.gamma_:.6f}")
        
        # Phase 3: Compute RBF activations
        rbf_matrix = self._compute_rbf_matrix(X_scaled, self.centers_)
        
        # Phase 4: Train output layer (supervised)
        self._solve_output_weights(rbf_matrix, y_encoded)
        
        if self.verbose:
            print(f"Trained output weights shape: {self.output_weights_.shape}")
        
        # Analyze the network
        if monitor_training:
            self._analyze_rbf_network(X_scaled, y_encoded, rbf_matrix)
        
        self.is_fitted_ = True
        return self
    
    def _analyze_rbf_network(self, X, y_encoded, rbf_matrix):
        """Analyze the trained RBF network"""
        # Center analysis
        center_distances = euclidean_distances(self.centers_, self.centers_)
        np.fill_diagonal(center_distances, np.inf)
        
        self.center_analysis_ = {
            "n_centers": len(self.centers_),
            "center_method": self.center_selection,
            "avg_center_distance": np.mean(center_distances[center_distances != np.inf]),
            "min_center_distance": np.min(center_distances[center_distances != np.inf]),
            "max_center_distance": np.max(center_distances[center_distances != np.inf]),
            "center_spread": np.std(center_distances[center_distances != np.inf])
        }
        
        # RBF activation analysis
        rbf_activations = rbf_matrix
        
        self.rbf_analysis_ = {
            "rbf_function": self.rbf_function,
            "gamma_parameter": self.gamma_,
            "avg_activation": np.mean(rbf_activations),
            "max_activation": np.max(rbf_activations),
            "min_activation": np.min(rbf_activations),
            "activation_std": np.std(rbf_activations),
            "sparsity": np.mean(rbf_activations < 0.1),  # Fraction of low activations
            "effective_centers": np.sum(np.max(rbf_activations, axis=0) > 0.1)  # Centers with significant activation
        }
        
        # Output weight analysis
        if hasattr(self, 'output_weights_'):
            weight_norms = np.linalg.norm(self.output_weights_, axis=1)
            self.rbf_analysis_["output_weight_stats"] = {
                "mean_weight_norm": np.mean(weight_norms),
                "max_weight_norm": np.max(weight_norms),
                "min_weight_norm": np.min(weight_norms),
                "weight_norm_std": np.std(weight_norms)
            }
    
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
        
        # Compute RBF activations
        rbf_matrix = self._compute_rbf_matrix(X_scaled, self.centers_)
        
        # Compute output
        output = rbf_matrix @ self.output_weights_
        
        # Convert to predictions
        if len(self.classes_) == 2:
            # Binary classification
            y_pred_encoded = (output.ravel() > 0).astype(int)
        else:
            # Multi-class classification
            y_pred_encoded = np.argmax(output, axis=1)
        
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
        
        # Compute RBF activations
        rbf_matrix = self._compute_rbf_matrix(X_scaled, self.centers_)
        
        # Compute output
        output = rbf_matrix @ self.output_weights_
        
        # Convert to probabilities
        if len(self.classes_) == 2:
            # Binary classification - apply sigmoid
            proba_positive = 1 / (1 + np.exp(-output.ravel()))
            probabilities = np.column_stack([1 - proba_positive, proba_positive])
        else:
            # Multi-class classification - apply softmax
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        
        return probabilities
    
    def get_rbf_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of RBF network structure and performance
        
        Returns:
        --------
        analysis_info : dict
            Comprehensive RBF network analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "network_summary": {
                "n_centers": len(self.centers_),
                "n_features": self.n_features_in_,
                "n_classes": len(self.classes_),
                "rbf_function": self.rbf_function,
                "gamma_parameter": self.gamma_,
                "center_selection": self.center_selection,
                "feature_scaling": self.scaler_ is not None
            },
            "architecture_details": {
                "input_layer_size": self.n_features_in_,
                "hidden_layer_size": len(self.centers_),
                "output_layer_size": len(self.classes_),
                "total_parameters": len(self.centers_) * len(self.classes_),
                "rbf_type": self.rbf_function,
                "center_method": self.center_selection
            }
        }
        
        # Add center analysis
        if self.center_analysis_:
            analysis["center_analysis"] = self.center_analysis_
        
        # Add RBF analysis
        if self.rbf_analysis_:
            analysis["rbf_analysis"] = self.rbf_analysis_
        
        return analysis
    
    def plot_rbf_analysis(self, figsize=(15, 10)):
        """
        Create comprehensive RBF network analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            RBF analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Center Distribution (2D projection if possible)
        if self.n_features_in_ >= 2:
            ax1.scatter(self.centers_[:, 0], self.centers_[:, 1], 
                       c='red', s=100, alpha=0.7, marker='x', linewidth=3, label='RBF Centers')
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.set_title('RBF Center Distribution (2D Projection)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.hist(self.centers_[:, 0], bins=min(20, len(self.centers_)//2), 
                    alpha=0.7, color='red', edgecolor='black')
            ax1.set_xlabel('Feature Value')
            ax1.set_ylabel('Number of Centers')
            ax1.set_title('RBF Center Distribution (1D)')
            ax1.grid(True, alpha=0.3)
        
        # 2. RBF Function Visualization
        distances = np.linspace(0, 3, 100)
        rbf_values = self._apply_rbf_function(distances)
        
        ax2.plot(distances, rbf_values, 'b-', linewidth=2, label=f'{self.rbf_function.capitalize()} RBF')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% Activation')
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='10% Activation')
        ax2.set_xlabel('Distance from Center')
        ax2.set_ylabel('RBF Activation')
        ax2.set_title(f'{self.rbf_function.capitalize()} RBF Function (γ={self.gamma_:.4f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Center Distance Analysis
        if len(self.centers_) > 1:
            center_distances = euclidean_distances(self.centers_, self.centers_)
            # Remove diagonal (self-distances)
            upper_triangle = np.triu(center_distances, k=1)
            pairwise_distances = upper_triangle[upper_triangle > 0]
            
            ax3.hist(pairwise_distances, bins=min(20, len(pairwise_distances)//2), 
                    alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(np.mean(pairwise_distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(pairwise_distances):.3f}')
            ax3.set_xlabel('Distance Between Centers')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Center-to-Center Distances')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Only one center\nNo distances to plot', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Center Distance Analysis')
        
        # 4. Output Weight Analysis
        if hasattr(self, 'output_weights_'):
            weight_norms = np.linalg.norm(self.output_weights_, axis=1)
            center_indices = range(len(weight_norms))
            
            bars = ax4.bar(center_indices, weight_norms, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_xlabel('RBF Center Index')
            ax4.set_ylabel('Output Weight Norm')
            ax4.set_title('Output Weight Magnitudes by Center')
            ax4.grid(True, alpha=0.3)
            
            # Highlight most important centers
            important_centers = np.argsort(weight_norms)[-3:]  # Top 3
            for i in important_centers:
                bars[i].set_color('orange')
            
            ax4.legend(['Standard Centers', 'Top 3 Important Centers'])
        else:
            ax4.text(0.5, 0.5, 'Output weights\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Output Weight Analysis')
        
        plt.tight_layout()
        return fig
    
    def plot_decision_boundary_2d(self, X, y, feature_indices=(0, 1), figsize=(10, 8), resolution=100):
        """
        Plot 2D decision boundary visualization
        
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
        
        # Create a temporary 2D RBF model
        temp_rbf = RBFNetworkPlugin(
            n_centers=self.n_centers,
            rbf_function=self.rbf_function,
            center_selection=self.center_selection,
            gamma=self.gamma,
            regularization=self.regularization,
            random_state=self.random_state,
            auto_scaling=False  # We handle scaling manually
        )
        
        # Encode labels
        y_encoded = self.label_encoder_.transform(y)
        temp_rbf.fit(X_2d_scaled, y_encoded)
        
        # Create meshgrid
        x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
        y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Predict on meshgrid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = temp_rbf.predict_proba(mesh_points)[:, 1] if len(self.classes_) == 2 else temp_rbf.predict(mesh_points)
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
        
        # Plot RBF centers
        if hasattr(temp_rbf, 'centers_') and temp_rbf.centers_.shape[1] == 2:
            ax.scatter(temp_rbf.centers_[:, 0], temp_rbf.centers_[:, 1], 
                      c='red', s=200, alpha=0.8, marker='x', linewidth=3, label='RBF Centers')
        
        ax.set_xlabel(f'Feature {feature_indices[0]}' + 
                     (f' ({self.feature_names_[feature_indices[0]]})' if self.feature_names_ else ''))
        ax.set_ylabel(f'Feature {feature_indices[1]}' + 
                     (f' ({self.feature_names_[feature_indices[1]]})' if self.feature_names_ else ''))
        ax.set_title(f'RBF Network Decision Boundary\n{self.n_centers} centers, {self.rbf_function} RBF')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🎯 RBF Network Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Architecture", "RBF Config", "Training", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Network Architecture**")
            
            # Number of centers
            n_centers = st.slider(
                "Number of RBF Centers:",
                min_value=1,
                max_value=100,
                value=int(self.n_centers),
                help="Number of radial basis function centers (hidden neurons)",
                key=f"{key_prefix}_n_centers"
            )
            
            # Center selection method
            center_selection = st.selectbox(
                "Center Selection Method:",
                options=['kmeans', 'random', 'data_points', 'grid'],
                index=['kmeans', 'random', 'data_points', 'grid'].index(self.center_selection),
                help="Method for positioning RBF centers",
                key=f"{key_prefix}_center_selection"
            )
            
            # RBF function type
            rbf_function = st.selectbox(
                "RBF Function Type:",
                options=['gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline'],
                index=['gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline'].index(self.rbf_function),
                help="Type of radial basis function",
                key=f"{key_prefix}_rbf_function"
            )
            
            # Display function info
            if rbf_function == 'gaussian':
                st.info("🔄 Gaussian: exp(-γ||x-c||²) - Most popular, smooth")
            elif rbf_function == 'multiquadric':
                st.info("📈 Multiquadric: √(||x-c||² + γ²) - Global support")
            elif rbf_function == 'inverse_multiquadric':
                st.info("📉 Inverse Multiquadric: 1/√(||x-c||² + γ²) - Bounded")
            elif rbf_function == 'thin_plate_spline':
                st.info("🌊 Thin Plate Spline: ||x-c||²log(||x-c||) - Smooth interpolation")
        
        with tab2:
            st.markdown("**RBF Configuration**")
            
            # Gamma parameter
            gamma_option = st.selectbox(
                "Gamma (Width) Parameter:",
                options=['auto', 'manual'],
                index=0 if self.gamma == 'auto' else 1,
                help="Controls the width/spread of RBF functions",
                key=f"{key_prefix}_gamma_option"
            )
            
            if gamma_option == 'manual':
                gamma = st.number_input(
                    "Manual Gamma Value:",
                    value=1.0 if self.gamma == 'auto' else float(self.gamma),
                    min_value=0.001,
                    max_value=100.0,
                    step=0.1,
                    format="%.3f",
                    help="Higher values = narrower RBF functions",
                    key=f"{key_prefix}_gamma_manual"
                )
            else:
                gamma = 'auto'
                st.info("Auto: γ based on average center distances")
            
            # Width optimization
            width_optimization = st.selectbox(
                "Width Optimization:",
                options=['heuristic', 'fixed', 'adaptive'],
                index=['heuristic', 'fixed', 'adaptive'].index(self.width_optimization),
                help="Method for determining RBF widths",
                key=f"{key_prefix}_width_optimization"
            )
            
            # Regularization
            regularization = st.number_input(
                "Regularization (L2):",
                value=float(self.regularization),
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                format="%.4f",
                help="L2 penalty for output weights (prevents overfitting)",
                key=f"{key_prefix}_regularization"
            )
            
            # Normalization options
            normalize_inputs = st.checkbox(
                "Normalize Inputs",
                value=self.normalize_inputs,
                help="Normalize input data",
                key=f"{key_prefix}_normalize_inputs"
            )
            
            normalize_outputs = st.checkbox(
                "Normalize Outputs",
                value=self.normalize_outputs,
                help="Normalize output data",
                key=f"{key_prefix}_normalize_outputs"
            )
        
        with tab3:
            st.markdown("**Training Configuration**")
            
            # Output layer solver
            output_layer_solver = st.selectbox(
                "Output Layer Solver:",
                options=['analytical', 'gradient'],
                index=['analytical', 'gradient'].index(self.output_layer_solver),
                help="Method for training output weights",
                key=f"{key_prefix}_output_layer_solver"
            )
            
            if output_layer_solver == 'gradient':
                # Learning rate
                learning_rate = st.number_input(
                    "Learning Rate:",
                    value=float(self.learning_rate),
                    min_value=0.001,
                    max_value=1.0,
                    step=0.001,
                    format="%.4f",
                    help="Learning rate for gradient descent",
                    key=f"{key_prefix}_learning_rate"
                )
                
                # Momentum
                momentum = st.slider(
                    "Momentum:",
                    min_value=0.0,
                    max_value=0.99,
                    value=float(self.momentum),
                    step=0.01,
                    help="Momentum for gradient descent",
                    key=f"{key_prefix}_momentum"
                )
            else:
                learning_rate = self.learning_rate
                momentum = self.momentum
            
            # Max iterations
            max_iter = st.slider(
                "Max Iterations:",
                min_value=50,
                max_value=1000,
                value=int(self.max_iter),
                help="Maximum training iterations",
                key=f"{key_prefix}_max_iter"
            )
            
            # Tolerance
            tol = st.number_input(
                "Tolerance:",
                value=float(self.tol),
                min_value=1e-6,
                max_value=1e-2,
                step=1e-4,
                format="%.6f",
                help="Convergence tolerance",
                key=f"{key_prefix}_tol"
            )
            
            # K-means specific parameters
            if center_selection == 'kmeans':
                st.markdown("**K-means Parameters:**")
                kmeans_init = st.selectbox(
                    "K-means Initialization:",
                    options=['k-means++', 'random'],
                    index=['k-means++', 'random'].index(self.kmeans_init),
                    help="K-means initialization method",
                    key=f"{key_prefix}_kmeans_init"
                )
            else:
                kmeans_init = self.kmeans_init
        
        with tab4:
            st.markdown("**Advanced Settings**")
            
            # Feature scaling
            auto_scaling = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scaling,
                help="Automatically scale features (recommended for RBF)",
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
            
            # Advanced optimization options
            adaptive_centers = st.checkbox(
                "Adaptive Centers",
                value=self.adaptive_centers,
                help="Allow center positions to adapt during training",
                key=f"{key_prefix}_adaptive_centers"
            )
            
            center_optimization = st.selectbox(
                "Center Optimization:",
                options=['gradient', 'none'],
                index=['gradient', 'none'].index(self.center_optimization),
                help="Method for optimizing center positions",
                key=f"{key_prefix}_center_optimization"
            )
            
            # Early stopping
            early_stopping = st.checkbox(
                "Early Stopping",
                value=self.early_stopping,
                help="Stop training when validation performance stops improving",
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
            else:
                validation_fraction = self.validation_fraction
            
            # Random state
            random_state = st.number_input(
                "Random Seed:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
            
            # Verbose
            verbose = st.checkbox(
                "Verbose Output",
                value=self.verbose,
                help="Print training progress",
                key=f"{key_prefix}_verbose"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if SKLEARN_AVAILABLE:
                st.success("✅ Required libraries are available")
            else:
                st.error("❌ scikit-learn not installed. Run: pip install scikit-learn")
            
            st.info("""
            **RBF Network** - Radial Basis Function Network:
            • 🎯 Three-layer architecture with RBF hidden layer
            • 📊 Universal function approximation capability
            • 🔄 Hybrid learning (unsupervised centers + supervised weights)
            • 📈 Excellent for function approximation and interpolation
            • 🚀 Fast training (analytical solution for output weights)
            • 🎪 Local response characteristics
            
            **Key Characteristics:**
            • Distance-based activation functions
            • Local learning and generalization
            • Interpretable center-based structure
            • Good interpolation properties
            """)
            
            # RBF function guide
            if st.button("🎯 RBF Function Guide", key=f"{key_prefix}_rbf_guide"):
                st.markdown("""
                **Radial Basis Function Types:**
                
                **Gaussian (Most Popular):**
                - φ(r) = exp(-γr²)
                - Smooth, infinitely differentiable
                - Local support, good generalization
                
                **Multiquadric:**
                - φ(r) = √(r² + γ²)
                - Global support, monotonic
                - Good for scattered data interpolation
                
                **Inverse Multiquadric:**
                - φ(r) = 1/√(r² + γ²)
                - Bounded, decreasing
                - Numerically stable
                
                **Thin Plate Spline:**
                - φ(r) = r²log(r)
                - Smooth interpolation
                - Natural for 2D problems
                """)
            
            # Center selection guide
            if st.button("🎪 Center Selection Guide", key=f"{key_prefix}_center_guide"):
                st.markdown("""
                **Center Selection Methods:**
                
                **K-means (Recommended):**
                - Data-driven center placement
                - Well-distributed centers
                - Good for most problems
                
                **Random:**
                - Fast, simple selection
                - May not be optimal
                - Good for quick prototyping
                
                **Data Points:**
                - Use training samples as centers
                - Guaranteed data coverage
                - Can lead to overfitting
                
                **Grid:**
                - Regular grid placement
                - Good for low dimensions
                - Suffers from curse of dimensionality
                """)
            
            # Parameter tuning guide
            if st.button("⚙️ Parameter Tuning Guide", key=f"{key_prefix}_tuning_guide"):
                st.markdown("""
                **RBF Network Parameter Tuning:**
                
                **Number of Centers:**
                - Start with √(n_samples)
                - More centers = more complexity
                - Balance between underfitting and overfitting
                
                **Gamma Parameter:**
                - Controls RBF width/spread
                - Auto: based on center distances
                - Manual: larger γ = narrower functions
                
                **Regularization:**
                - Prevents overfitting in output layer
                - Use when centers >> samples
                - Start with 0, increase if overfitting
                
                **Center Selection:**
                - K-means: best general choice
                - Random: for quick experiments
                - Data points: for small datasets
                """)
            
            # Algorithm comparison
            if st.button("📊 vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **RBF vs Other Neural Networks:**
                
                **vs MLP:**
                - RBF: Fixed 3-layer, MLP: Variable layers
                - RBF: Radial activation, MLP: Sigmoid/ReLU
                - RBF: Faster training, MLP: More flexible
                - RBF: Local response, MLP: Global response
                
                **vs SVM with RBF Kernel:**
                - Similar mathematical foundation
                - RBF Network: Regression-like output
                - SVM: Margin-based classification
                - RBF: Faster prediction, SVM: Better generalization
                
                **vs K-NN:**
                - Both use distance-based decisions
                - RBF: Parametric model, K-NN: Non-parametric
                - RBF: Faster prediction, K-NN: Simpler concept
                """)
        
        return {
            "n_centers": n_centers,
            "rbf_function": rbf_function,
            "center_selection": center_selection,
            "gamma": gamma,
            "regularization": regularization,
            "normalize_inputs": normalize_inputs,
            "normalize_outputs": normalize_outputs,
            "random_state": random_state,
            "kmeans_init": kmeans_init,
            "max_iter": max_iter,
            "tol": tol,
            "adaptive_centers": adaptive_centers,
            "center_optimization": center_optimization,
            "width_optimization": width_optimization,
            "output_layer_solver": output_layer_solver,
            "early_stopping": early_stopping,
            "validation_fraction": validation_fraction,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "auto_scaling": auto_scaling,
            "scaling_method": scaling_method,
            "verbose": verbose
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return RBFNetworkPlugin(
            n_centers=hyperparameters.get("n_centers", self.n_centers),
            rbf_function=hyperparameters.get("rbf_function", self.rbf_function),
            center_selection=hyperparameters.get("center_selection", self.center_selection),
            gamma=hyperparameters.get("gamma", self.gamma),
            regularization=hyperparameters.get("regularization", self.regularization),
            normalize_inputs=hyperparameters.get("normalize_inputs", self.normalize_inputs),
            normalize_outputs=hyperparameters.get("normalize_outputs", self.normalize_outputs),
            random_state=hyperparameters.get("random_state", self.random_state),
            kmeans_init=hyperparameters.get("kmeans_init", self.kmeans_init),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            tol=hyperparameters.get("tol", self.tol),
            adaptive_centers=hyperparameters.get("adaptive_centers", self.adaptive_centers),
            center_optimization=hyperparameters.get("center_optimization", self.center_optimization),
            width_optimization=hyperparameters.get("width_optimization", self.width_optimization),
            output_layer_solver=hyperparameters.get("output_layer_solver", self.output_layer_solver),
            early_stopping=hyperparameters.get("early_stopping", self.early_stopping),
            validation_fraction=hyperparameters.get("validation_fraction", self.validation_fraction),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            momentum=hyperparameters.get("momentum", self.momentum),
            auto_scaling=hyperparameters.get("auto_scaling", self.auto_scaling),
            scaling_method=hyperparameters.get("scaling_method", self.scaling_method),
            verbose=hyperparameters.get("verbose", self.verbose)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for RBF Network
        
        RBF networks benefit from feature scaling and don't handle missing values well.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # Check for missing values
        if np.any(pd.isna(X_processed)):
            warnings.warn("RBF networks don't handle missing values. Consider imputation before training.")
        
        # Check for infinite values
        if np.any(np.isinf(X_processed)):
            warnings.warn("RBF networks don't handle infinite values. Please clean your data.")
        
# Continue from line 1438 where the code breaks:
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if RBF Network is compatible with the given data
        
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
            return False, f"RBF Network requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for missing values
        if np.any(pd.isna(X)):
            return False, "RBF Network doesn't handle missing values well. Please impute missing values first."
        
        # Check for infinite values
        if np.any(np.isinf(X)):
            return False, "RBF Network doesn't handle infinite values. Please clean your data first."
        
        # Check number of centers vs samples
        if self.n_centers > X.shape[0]:
            return True, f"Warning: More centers ({self.n_centers}) than samples ({X.shape[0]}). Consider reducing centers."
        
        # Check dimensionality
        if X.shape[1] > 20:
            return True, "Warning: High dimensionality may affect RBF performance. Consider dimensionality reduction."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            # Check class distribution
            class_counts = np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes)
            min_class_size = np.min(class_counts)
            if min_class_size < 5:
                return True, f"Warning: Very small class detected ({min_class_size} samples). May affect performance."
        
        return True, "RBF Network is compatible with this data. Consider feature scaling for best results!"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "n_centers": len(self.centers_) if self.centers_ is not None else None,
            "rbf_function": self.rbf_function,
            "center_selection": self.center_selection,
            "gamma_parameter": getattr(self, 'gamma_', None),
            "regularization": self.regularization,
            "scaling_applied": self.scaler_ is not None,
            "scaling_method": self.scaling_method if self.scaler_ is not None else None,
            "output_layer_solver": self.output_layer_solver,
            "total_parameters": len(self.centers_) * len(self.classes_) if self.centers_ is not None and self.classes_ is not None else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "RBF Network",
            "training_completed": True,
            "rbf_network_characteristics": {
                "three_layer_architecture": True,
                "radial_basis_functions": True,
                "local_approximation": True,
                "universal_approximator": True,
                "hybrid_learning": True,
                "center_based": True
            },
            "model_configuration": {
                "n_centers": len(self.centers_) if self.centers_ is not None else None,
                "rbf_function": self.rbf_function,
                "center_selection_method": self.center_selection,
                "gamma_parameter": getattr(self, 'gamma_', None),
                "regularization": self.regularization,
                "feature_scaling": self.scaler_ is not None,
                "output_solver": self.output_layer_solver
            },
            "rbf_analysis": self.get_rbf_analysis(),
            "performance_considerations": {
                "memory_usage": "Moderate - stores centers and weights",
                "prediction_time": "Fast - RBF computation + linear combination",
                "training_time": "Fast - analytical solution for output weights",
                "scalability": "Good for small to medium datasets",
                "hyperparameter_sensitivity": "Moderate - center placement and gamma critical",
                "dimensionality_sensitivity": "High - curse of dimensionality"
            },
            "mathematical_properties": {
                "function_approximation": "Universal approximation capability",
                "interpolation": "Excellent interpolation properties",
                "locality": "Local response - changes affect nearby regions",
                "smoothness": "Smooth basis functions ensure smooth output",
                "center_interpretation": "Centers represent important regions in input space"
            }
        }
        
        return info
    
    def get_rbf_centers_info(self) -> Dict[str, Any]:
        """Get detailed information about RBF centers"""
        if not self.is_fitted_ or self.centers_ is None:
            return {"status": "Not fitted or no centers available"}
        
        # Analyze center distribution
        center_distances = euclidean_distances(self.centers_, self.centers_)
        np.fill_diagonal(center_distances, np.inf)
        
        # Find nearest neighbors for each center
        nearest_distances = np.min(center_distances, axis=1)
        
        # Analyze center coverage
        if hasattr(self, 'rbf_analysis_') and 'effective_centers' in self.rbf_analysis_:
            effective_centers = self.rbf_analysis_['effective_centers']
        else:
            effective_centers = len(self.centers_)
        
        center_info = {
            "total_centers": len(self.centers_),
            "effective_centers": effective_centers,
            "center_efficiency": effective_centers / len(self.centers_),
            "center_selection_method": self.center_selection,
            "center_statistics": {
                "avg_nearest_distance": np.mean(nearest_distances),
                "min_nearest_distance": np.min(nearest_distances),
                "max_nearest_distance": np.max(nearest_distances),
                "distance_std": np.std(nearest_distances)
            },
            "rbf_parameters": {
                "function_type": self.rbf_function,
                "gamma_parameter": getattr(self, 'gamma_', None),
                "width_optimization": self.width_optimization
            }
        }
        
        # Add center coordinates summary
        if self.centers_.shape[1] <= 10:  # Only for reasonable dimensions
            center_info["center_coordinates"] = {
                "mean_position": np.mean(self.centers_, axis=0).tolist(),
                "std_position": np.std(self.centers_, axis=0).tolist(),
                "min_position": np.min(self.centers_, axis=0).tolist(),
                "max_position": np.max(self.centers_, axis=0).tolist()
            }
        
        return center_info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the RBF Network model.

        These metrics are derived from the internal analyses performed during fit()
        (e.g., center distribution, RBF activation characteristics).
        Parameters y_true, y_pred, y_proba are kept for API consistency but are not
        directly used as metrics are sourced from internal analysis attributes.

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
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve RBF Network specific metrics."}

        metrics = {}
        prefix = "rbf_" # Prefix for RBF Network specific metrics

        # From self.center_analysis_
        if hasattr(self, 'center_analysis_') and self.center_analysis_:
            if 'avg_center_distance' in self.center_analysis_:
                metrics[f"{prefix}avg_center_distance"] = self.center_analysis_['avg_center_distance']
            if 'min_center_distance' in self.center_analysis_:
                metrics[f"{prefix}min_center_distance"] = self.center_analysis_['min_center_distance']
            if 'max_center_distance' in self.center_analysis_:
                metrics[f"{prefix}max_center_distance"] = self.center_analysis_['max_center_distance']
            if 'center_spread' in self.center_analysis_:
                metrics[f"{prefix}center_spread_std"] = self.center_analysis_['center_spread']
            if 'n_centers' in self.center_analysis_: # Actual number of centers used
                metrics[f"{prefix}num_centers_used"] = self.center_analysis_['n_centers']


        # From self.rbf_analysis_
        if hasattr(self, 'rbf_analysis_') and self.rbf_analysis_:
            if 'gamma_parameter' in self.rbf_analysis_:
                metrics[f"{prefix}gamma_parameter_used"] = self.rbf_analysis_['gamma_parameter']
            if 'avg_activation' in self.rbf_analysis_:
                metrics[f"{prefix}avg_rbf_activation"] = self.rbf_analysis_['avg_activation']
            if 'activation_std' in self.rbf_analysis_:
                metrics[f"{prefix}std_rbf_activation"] = self.rbf_analysis_['activation_std']
            if 'sparsity' in self.rbf_analysis_: # Fraction of low activations
                metrics[f"{prefix}rbf_activation_sparsity_lt_0_1"] = self.rbf_analysis_['sparsity']
            if 'effective_centers' in self.rbf_analysis_:
                metrics[f"{prefix}num_effective_centers"] = self.rbf_analysis_['effective_centers']
            
            output_weights_stats = self.rbf_analysis_.get("output_weight_stats", {})
            if 'mean_weight_norm' in output_weights_stats:
                metrics[f"{prefix}mean_output_weight_norm"] = output_weights_stats['mean_weight_norm']
            if 'max_weight_norm' in output_weights_stats:
                metrics[f"{prefix}max_output_weight_norm"] = output_weights_stats['max_weight_norm']
            if 'min_weight_norm' in output_weights_stats:
                metrics[f"{prefix}min_output_weight_norm"] = output_weights_stats['min_weight_norm']
        
        if not metrics:
            metrics['info'] = "No specific RBF Network metrics were available from internal analyses."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return RBFNetworkPlugin()
