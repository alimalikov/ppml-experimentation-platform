"""
Multi-layer Perceptron Regressor Core Implementation
===================================================

This module provides the core neural network functionality for the MLP Regressor,
including network training, architecture management, and advanced neural network
analysis capabilities.

Features:
- Flexible network architecture configuration
- Multiple solver options (LBFGS, SGD, Adam)
- Advanced training monitoring and control
- Weight and bias analysis
- Network performance optimization
- Training process visualization support

Author: Bachelor Thesis Project
Date: June 2025
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
import logging
import time
from copy import deepcopy

# Core ML libraries
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPCore:
    """
    Core Multi-layer Perceptron Regressor implementation with advanced functionality.
    
    This class provides comprehensive neural network capabilities including:
    - Flexible architecture configuration
    - Multiple training strategies
    - Advanced monitoring and analysis
    - Performance optimization features
    """
    
    def __init__(self,
                 hidden_layer_sizes: Tuple[int, ...] = (100,),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 batch_size: Union[int, str] = 'auto',
                 learning_rate: str = 'constant',
                 learning_rate_init: float = 0.001,
                 power_t: float = 0.5,
                 max_iter: int = 200,
                 shuffle: bool = True,
                 random_state: Optional[int] = None,
                 tol: float = 1e-4,
                 verbose: bool = False,
                 warm_start: bool = False,
                 momentum: float = 0.9,
                 nesterovs_momentum: bool = True,
                 early_stopping: bool = False,
                 validation_fraction: float = 0.1,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8,
                 n_iter_no_change: int = 10,
                 max_fun: int = 15000,
                 auto_scale: bool = True,
                 scaler_type: str = 'standard'):
        """
        Initialize the MLP Core with comprehensive configuration options.
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple of int, default=(100,)
            The ith element represents the number of neurons in the ith hidden layer
        activation : str, default='relu'
            Activation function for the hidden layer ('identity', 'logistic', 'tanh', 'relu')
        solver : str, default='adam'
            The solver for weight optimization ('lbfgs', 'sgd', 'adam')
        alpha : float, default=0.0001
            L2 penalty (regularization term) parameter
        batch_size : int or 'auto', default='auto'
            Size of minibatches for stochastic optimizers
        learning_rate : str, default='constant'
            Learning rate schedule for weight updates ('constant', 'invscaling', 'adaptive')
        learning_rate_init : float, default=0.001
            The initial learning rate used
        power_t : float, default=0.5
            The exponent for inverse scaling learning rate
        max_iter : int, default=200
            Maximum number of iterations
        shuffle : bool, default=True
            Whether to shuffle samples in each iteration
        random_state : int, optional
            Random state for reproducibility
        tol : float, default=1e-4
            Tolerance for the optimization
        verbose : bool, default=False
            Whether to print progress messages
        warm_start : bool, default=False
            Whether to reuse the solution of the previous call to fit
        momentum : float, default=0.9
            Momentum for gradient descent update
        nesterovs_momentum : bool, default=True
            Whether to use Nesterov's momentum
        early_stopping : bool, default=False
            Whether to use early stopping to terminate training
        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set
        beta_1 : float, default=0.9
            Exponential decay rate for estimates of first moment vector in adam
        beta_2 : float, default=0.999
            Exponential decay rate for estimates of second moment vector in adam
        epsilon : float, default=1e-8
            Value for numerical stability in adam
        n_iter_no_change : int, default=10
            Maximum number of epochs to not meet tol improvement
        max_fun : int, default=15000
            Maximum number of loss function calls
        auto_scale : bool, default=True
            Whether to automatically scale features
        scaler_type : str, default='standard'
            Type of scaler to use ('standard', 'minmax', 'robust')
        """
        
        # Network architecture parameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        
        # Training parameters
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
        
        # Optimization parameters
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        
        # Preprocessing parameters
        self.auto_scale = auto_scale
        self.scaler_type = scaler_type
        
        # Internal state
        self.model_ = None
        self.scaler_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.training_history_ = {}
        self.is_fitted_ = False
        
        # Performance tracking
        self.training_time_ = 0.0
        self.training_score_ = 0.0
        self.validation_score_ = 0.0
        
        logger.info("✅ MLP Core initialized successfully")
    
    def _create_model(self) -> MLPRegressor:
        """Create MLPRegressor with current parameters."""
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
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
    
    def _create_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        """Create scaler based on scaler_type."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            logger.warning(f"⚠️ Unknown scaler type '{self.scaler_type}', using StandardScaler")
            return StandardScaler()
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate input data for training."""
        try:
            # Basic shape validation
            if X.shape[0] != y.shape[0]:
                return {
                    'valid': False,
                    'error': f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}"
                }
            
            # Check for minimum samples
            min_samples = max(10, sum(self.hidden_layer_sizes) + 1)
            if X.shape[0] < min_samples:
                return {
                    'valid': False,
                    'error': f"Insufficient samples for network architecture. Need at least {min_samples}, got {X.shape[0]}"
                }
            
            # Check for missing values
            if np.isnan(X).any() or np.isnan(y).any():
                return {
                    'valid': False,
                    'error': "Input data contains NaN values"
                }
            
            # Check for infinite values
            if np.isinf(X).any() or np.isinf(y).any():
                return {
                    'valid': False,
                    'error': "Input data contains infinite values"
                }
            
            # Network architecture validation
            if not self.hidden_layer_sizes or any(size <= 0 for size in self.hidden_layer_sizes):
                return {
                    'valid': False,
                    'error': "Hidden layer sizes must be positive integers"
                }
            
            # Memory estimation for large networks
            total_params = self._estimate_parameters(X.shape[1])
            memory_mb = (total_params * 8) / (1024 * 1024)  # 8 bytes per float64
            
            if memory_mb > 1000:  # 1GB threshold
                logger.warning(f"⚠️ Large network detected: ~{memory_mb:.1f}MB memory required")
            
            return {
                'valid': True,
                'samples': X.shape[0],
                'features': X.shape[1],
                'estimated_parameters': total_params,
                'estimated_memory_mb': memory_mb
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation failed: {str(e)}"
            }
    
    def _estimate_parameters(self, n_features: int) -> int:
        """Estimate total number of parameters in the network."""
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        total_params = 0
        
        for i in range(len(layer_sizes) - 1):
            # Weights + biases
            total_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        
        return total_params
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the MLP model with comprehensive monitoring.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metadata
        """
        try:
            logger.info("🚀 Starting MLP neural network training...")
            
            # Validate inputs
            validation_result = self._validate_inputs(X, y)
            if not validation_result['valid']:
                return {'model_fitted': False, 'error': validation_result['error']}
            
            # Store training data
            self.X_train_ = X.copy()
            self.y_train_ = y.copy()
            
            # Preprocessing
            if self.auto_scale:
                self.scaler_ = self._create_scaler()
                X_scaled = self.scaler_.fit_transform(X)
                logger.info(f"✅ Features scaled using {self.scaler_type} scaler")
            else:
                X_scaled = X
                logger.info("ℹ️ Feature scaling disabled")
            
            # Create and train model
            start_time = time.time()
            self.model_ = self._create_model()
            
            # Enhanced training with monitoring
            if self.verbose:
                logger.info(f"🏗️ Network architecture: {[X.shape[1]] + list(self.hidden_layer_sizes) + [1]}")
                logger.info(f"🎯 Total parameters: {validation_result['estimated_parameters']:,}")
                logger.info(f"🔧 Solver: {self.solver}, Activation: {self.activation}")
            
            # Train the model
            self.model_.fit(X_scaled, y)
            self.training_time_ = time.time() - start_time
            
            # Training success validation
            if not hasattr(self.model_, 'coefs_'):
                return {
                    'model_fitted': False,
                    'error': 'Model training failed - no coefficients found'
                }
            
            # Calculate training performance
            y_pred_train = self.model_.predict(X_scaled)
            self.training_score_ = r2_score(y, y_pred_train)
            
            # Store training history
            self.training_history_ = {
                'loss_curve': getattr(self.model_, 'loss_curve_', []),
                'validation_scores': getattr(self.model_, 'validation_scores_', []),
                'n_iter': getattr(self.model_, 'n_iter_', self.max_iter),
                'n_layers': getattr(self.model_, 'n_layers_', len(self.hidden_layer_sizes) + 2),
                'n_outputs': getattr(self.model_, 'n_outputs_', 1),
                'out_activation': getattr(self.model_, 'out_activation_', 'identity')
            }
            
            self.is_fitted_ = True
            
            # Training summary
            training_summary = {
                'model_fitted': True,
                'training_time': self.training_time_,
                'training_r2': self.training_score_,
                'final_loss': self.training_history_['loss_curve'][-1] if self.training_history_['loss_curve'] else None,
                'n_iterations': self.training_history_['n_iter'],
                'converged': self.training_history_['n_iter'] < self.max_iter,
                'network_info': {
                    'architecture': [X.shape[1]] + list(self.hidden_layer_sizes) + [1],
                    'total_parameters': validation_result['estimated_parameters'],
                    'n_layers': self.training_history_['n_layers'],
                    'activation': self.activation,
                    'solver': self.solver
                }
            }
            
            logger.info(f"✅ Training completed in {self.training_time_:.2f}s")
            logger.info(f"📊 Training R²: {self.training_score_:.4f}")
            logger.info(f"🔄 Iterations: {self.training_history_['n_iter']}/{self.max_iter}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return {
                'model_fitted': False,
                'error': str(e),
                'training_time': getattr(self, 'training_time_', 0.0)
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply same preprocessing as training
        if self.auto_scale and self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        return self.model_.predict(X_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score on given data."""
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def get_network_weights(self) -> List[np.ndarray]:
        """Get all network weights."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before accessing weights")
        return deepcopy(self.model_.coefs_)
    
    def get_network_biases(self) -> List[np.ndarray]:
        """Get all network biases."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before accessing biases")
        return deepcopy(self.model_.intercepts_)
    
    def get_network_architecture(self) -> Dict[str, Any]:
        """Get detailed network architecture information."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before accessing architecture")
        
        weights = self.get_network_weights()
        biases = self.get_network_biases()
        
        architecture = {
            'input_size': weights[0].shape[0],
            'hidden_layers': [],
            'output_size': weights[-1].shape[1],
            'total_parameters': 0,
            'layer_details': []
        }
        
        # Analyze each layer
        for i, (w, b) in enumerate(zip(weights, biases)):
            layer_info = {
                'layer_index': i,
                'input_size': w.shape[0],
                'output_size': w.shape[1],
                'weights_shape': w.shape,
                'biases_shape': b.shape,
                'parameters': w.size + b.size,
                'weight_stats': {
                    'mean': float(np.mean(w)),
                    'std': float(np.std(w)),
                    'min': float(np.min(w)),
                    'max': float(np.max(w))
                },
                'bias_stats': {
                    'mean': float(np.mean(b)),
                    'std': float(np.std(b)),
                    'min': float(np.min(b)),
                    'max': float(np.max(b))
                }
            }
            
            if i < len(weights) - 1:  # Hidden layer
                architecture['hidden_layers'].append(w.shape[1])
            
            architecture['layer_details'].append(layer_info)
            architecture['total_parameters'] += layer_info['parameters']
        
        return architecture
    
    def analyze_training_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence patterns."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before analyzing convergence")
        
        loss_curve = self.training_history_['loss_curve']
        if not loss_curve:
            return {'error': 'No loss curve available'}
        
        analysis = {
            'total_iterations': len(loss_curve),
            'final_loss': loss_curve[-1],
            'initial_loss': loss_curve[0],
            'loss_reduction': loss_curve[0] - loss_curve[-1],
            'loss_reduction_ratio': (loss_curve[0] - loss_curve[-1]) / loss_curve[0] if loss_curve[0] != 0 else 0,
            'converged': self.training_history_['n_iter'] < self.max_iter,
            'convergence_rate': 'fast' if len(loss_curve) < self.max_iter * 0.3 else 
                              'moderate' if len(loss_curve) < self.max_iter * 0.7 else 'slow'
        }
        
        # Analyze convergence stability
        if len(loss_curve) > 10:
            recent_losses = loss_curve[-10:]
            loss_variance = np.var(recent_losses)
            analysis['stability'] = 'stable' if loss_variance < 1e-6 else 'unstable'
            analysis['recent_variance'] = float(loss_variance)
        
        # Detect potential issues
        issues = []
        if not analysis['converged']:
            issues.append('did_not_converge')
        if analysis['loss_reduction_ratio'] < 0.01:
            issues.append('minimal_improvement')
        if len(loss_curve) > 1 and loss_curve[-1] > loss_curve[-2]:
            issues.append('loss_increasing')
        
        analysis['potential_issues'] = issues
        
        return analysis
    
    def get_activation_patterns(self, X: np.ndarray, layer_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze activation patterns in hidden layers.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data to analyze
        layer_index : int, optional
            Specific layer to analyze (if None, analyzes all hidden layers)
            
        Returns:
        --------
        Dict[str, Any]
            Activation pattern analysis
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before analyzing activations")
        
        # Apply preprocessing
        if self.auto_scale and self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get network weights and biases
        weights = self.get_network_weights()
        biases = self.get_network_biases()
        
        activations = {}
        current_activation = X_scaled
        
        # Forward pass through each layer
        for i, (w, b) in enumerate(zip(weights, biases)):
            # Linear transformation
            linear_output = np.dot(current_activation, w) + b
            
            # Apply activation function (except for output layer)
            if i < len(weights) - 1:  # Hidden layers
                if self.activation == 'relu':
                    current_activation = np.maximum(0, linear_output)
                elif self.activation == 'tanh':
                    current_activation = np.tanh(linear_output)
                elif self.activation == 'logistic':
                    current_activation = 1 / (1 + np.exp(-np.clip(linear_output, -500, 500)))
                elif self.activation == 'identity':
                    current_activation = linear_output
                
                # Store activation statistics
                activations[f'layer_{i}'] = {
                    'linear_output_stats': {
                        'mean': float(np.mean(linear_output)),
                        'std': float(np.std(linear_output)),
                        'min': float(np.min(linear_output)),
                        'max': float(np.max(linear_output))
                    },
                    'activation_stats': {
                        'mean': float(np.mean(current_activation)),
                        'std': float(np.std(current_activation)),
                        'min': float(np.min(current_activation)),
                        'max': float(np.max(current_activation))
                    },
                    'dead_neurons': int(np.sum(np.all(current_activation == 0, axis=0))),
                    'active_neurons': int(np.sum(np.any(current_activation != 0, axis=0))),
                    'activation_density': float(np.mean(current_activation != 0))
                }
            else:  # Output layer
                activations['output'] = {
                    'predictions': current_activation,
                    'stats': {
                        'mean': float(np.mean(linear_output)),
                        'std': float(np.std(linear_output)),
                        'min': float(np.min(linear_output)),
                        'max': float(np.max(linear_output))
                    }
                }
        
        # Return specific layer or all layers
        if layer_index is not None:
            layer_key = f'layer_{layer_index}'
            if layer_key in activations:
                return {layer_key: activations[layer_key]}
            else:
                return {'error': f'Layer {layer_index} not found'}
        
        return activations
    
    def suggest_architecture(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Suggest optimal network architecture based on data characteristics.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets
            
        Returns:
        --------
        Dict[str, Any]
            Architecture suggestions and reasoning
        """
        n_samples, n_features = X.shape
        
        suggestions = {
            'current_architecture': list(self.hidden_layer_sizes),
            'recommendations': []
        }
        
        # Rule-based suggestions
        
        # 1. Based on sample size
        if n_samples < 100:
            suggestions['recommendations'].append({
                'reason': 'small_dataset',
                'suggestion': 'Use smaller network to prevent overfitting',
                'recommended_sizes': (min(50, n_features),),
                'confidence': 'high'
            })
        elif n_samples > 10000:
            suggestions['recommendations'].append({
                'reason': 'large_dataset',
                'suggestion': 'Can use larger network for more capacity',
                'recommended_sizes': (min(200, n_features * 2), min(100, n_features)),
                'confidence': 'medium'
            })
        
        # 2. Based on feature count
        if n_features > 100:
            suggestions['recommendations'].append({
                'reason': 'high_dimensional',
                'suggestion': 'Use dimensionality reduction in hidden layers',
                'recommended_sizes': (n_features // 2, n_features // 4),
                'confidence': 'medium'
            })
        elif n_features < 10:
            suggestions['recommendations'].append({
                'reason': 'low_dimensional',
                'suggestion': 'Simple architecture sufficient',
                'recommended_sizes': (max(10, n_features * 2),),
                'confidence': 'high'
            })
        
        # 3. Based on complexity estimation
        y_std = np.std(y)
        if y_std < np.mean(np.abs(y)) * 0.1:  # Low variance target
            suggestions['recommendations'].append({
                'reason': 'simple_target',
                'suggestion': 'Target appears simple, small network recommended',
                'recommended_sizes': (n_features,),
                'confidence': 'medium'
            })
        
        # 4. Current architecture analysis
        current_params = self._estimate_parameters(n_features)
        param_to_sample_ratio = current_params / n_samples
        
        if param_to_sample_ratio > 1.0:
            suggestions['recommendations'].append({
                'reason': 'overparameterized',
                'suggestion': 'Current network may be too large for dataset',
                'recommended_sizes': (max(10, n_features // 2),),
                'confidence': 'high'
            })
        elif param_to_sample_ratio < 0.1:
            suggestions['recommendations'].append({
                'reason': 'underparameterized',
                'suggestion': 'Current network may be too small',
                'recommended_sizes': (n_features * 2, n_features),
                'confidence': 'medium'
            })
        
        # Generate overall recommendation
        if suggestions['recommendations']:
            # Use the highest confidence recommendation
            best_rec = max(suggestions['recommendations'], key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['confidence']])
            suggestions['best_recommendation'] = best_rec['recommended_sizes']
            suggestions['primary_reason'] = best_rec['reason']
        else:
            suggestions['best_recommendation'] = self.hidden_layer_sizes
            suggestions['primary_reason'] = 'current_architecture_suitable'
        
        return suggestions
    
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray, method: str = 'permutation') -> Dict[str, Any]:
        """
        Calculate feature importance for the neural network.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        method : str, default='permutation'
            Method to calculate importance ('permutation', 'weights')
            
        Returns:
        --------
        Dict[str, Any]
            Feature importance analysis
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calculating feature importance")
        
        if method == 'permutation':
            return self._permutation_importance(X, y)
        elif method == 'weights':
            return self._weight_based_importance()
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _permutation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate permutation-based feature importance."""
        baseline_score = self.score(X, y)
        importances = []
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            # Permute feature i
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self.score(X_permuted, y)
            importance = baseline_score - permuted_score
            importances.append(importance)
        
        importances = np.array(importances)
        
        return {
            'method': 'permutation',
            'baseline_score': baseline_score,
            'importances': importances.tolist(),
            'normalized_importances': (importances / np.sum(np.abs(importances))).tolist() if np.sum(np.abs(importances)) > 0 else importances.tolist(),
            'feature_ranking': np.argsort(importances)[::-1].tolist()
        }
    
    def _weight_based_importance(self) -> Dict[str, Any]:
        """Calculate weight-based feature importance (first layer weights)."""
        weights = self.get_network_weights()
        first_layer_weights = weights[0]  # Shape: (n_features, n_hidden)
        
        # Calculate importance as sum of absolute weights
        importances = np.sum(np.abs(first_layer_weights), axis=1)
        normalized_importances = importances / np.sum(importances) if np.sum(importances) > 0 else importances
        
        return {
            'method': 'weights',
            'importances': importances.tolist(),
            'normalized_importances': normalized_importances.tolist(),
            'feature_ranking': np.argsort(importances)[::-1].tolist()
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get all model parameters."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'learning_rate_init': self.learning_rate_init,
            'power_t': self.power_t,
            'max_iter': self.max_iter,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'tol': self.tol,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'momentum': self.momentum,
            'nesterovs_momentum': self.nesterovs_momentum,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'n_iter_no_change': self.n_iter_no_change,
            'max_fun': self.max_fun,
            'auto_scale': self.auto_scale,
            'scaler_type': self.scaler_type
        }
    
    def clone_with_params(self, **new_params) -> 'MLPCore':
        """Create a copy of this MLPCore with modified parameters."""
        current_params = self.get_model_params()
        current_params.update(new_params)
        return MLPCore(**current_params)