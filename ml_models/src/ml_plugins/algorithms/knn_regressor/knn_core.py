"""
K-Nearest Neighbors Regressor Core Implementation
=================================================

This module provides the core implementation of the K-Nearest Neighbors Regressor
algorithm with advanced features for analysis and optimization.

Features:
- Multiple distance metrics and algorithms
- Automatic K optimization
- Neighbor analysis capabilities
- Performance profiling
- Comprehensive parameter management

Author: Bachelor Thesis Project
Date: June 2025
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, List, Tuple, Union, Optional
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils.validation import check_X_y, check_array
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNNCore:
    """
    Core K-Nearest Neighbors Regressor implementation with advanced features.
    
    This class provides the fundamental KNN regression capabilities along with
    advanced analysis features for parameter optimization and performance evaluation.
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 weights: str = 'uniform',
                 algorithm: str = 'auto',
                 metric: str = 'euclidean',
                 p: int = 2,
                 auto_scale: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize the KNN Core component.
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to use for prediction
        weights : str, default='uniform'
            Weight function ('uniform', 'distance')
        algorithm : str, default='auto'
            Algorithm for neighbor search ('auto', 'ball_tree', 'kd_tree', 'brute')
        metric : str, default='euclidean'
            Distance metric ('euclidean', 'manhattan', 'chebyshev', 'minkowski')
        p : int, default=2
            Parameter for Minkowski metric
        auto_scale : bool, default=True
            Whether to automatically scale features
        random_state : int, optional
            Random state for reproducibility
        """
        
        # Core parameters
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.auto_scale = auto_scale
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
        # Training data storage for analysis
        self.X_train_ = None
        self.y_train_ = None
        self.X_train_scaled_ = None
        
        # Analysis results cache
        self._analysis_cache = {}
        
        # Performance tracking
        self.training_time_ = None
        self.prediction_times_ = []
        
        # Configuration for analysis components
        self.config = {
            'cv_folds': 5,
            'k_range': (1, 20),
            'optimization_metric': 'r2',
            'n_jobs': 1,
            'verbose': False
        }
        
        logger.info(f"✅ KNN Core initialized with n_neighbors={n_neighbors}, metric={metric}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the core component with advanced settings.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary with analysis settings
        """
        try:
            self.config.update(config)
            
            # Update model parameters if provided
            if 'n_neighbors' in config:
                self.n_neighbors = config['n_neighbors']
            if 'weights' in config:
                self.weights = config['weights']
            if 'algorithm' in config:
                self.algorithm = config['algorithm']
            if 'metric' in config:
                self.metric = config['metric']
            if 'p' in config:
                self.p = config['p']
            if 'auto_scale' in config:
                self.auto_scale = config['auto_scale']
            
            # Recreate model if fitted and parameters changed
            if self.is_fitted and any(param in config for param in 
                                    ['n_neighbors', 'weights', 'algorithm', 'metric', 'p']):
                self._create_model()
                
            logger.info("✅ KNN Core configuration updated")
            
        except Exception as e:
            logger.error(f"❌ Configuration failed: {str(e)}")
            raise
    
    def _create_model(self) -> KNeighborsRegressor:
        """Create KNN model instance with current parameters."""
        try:
            model_params = {
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'algorithm': self.algorithm,
                'metric': self.metric,
                'n_jobs': self.config.get('n_jobs', 1)
            }
            
            # Add p parameter only for minkowski metric
            if self.metric == 'minkowski':
                model_params['p'] = self.p
            
            self.model = KNeighborsRegressor(**model_params)
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Model creation failed: {str(e)}")
            raise
    
    def _create_scaler(self) -> Union[StandardScaler, MinMaxScaler]:
        """Create feature scaler."""
        if self.auto_scale:
            # Use StandardScaler for most cases, MinMaxScaler for specific metrics
            if self.metric in ['chebyshev']:
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
        else:
            self.scaler = None
        
        return self.scaler
    
    def _validate_and_prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and prepare data for training/prediction."""
        try:
            # Validate input data
            if y is not None:
                X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
            else:
                X = check_array(X, accept_sparse=False)
            
            # Check minimum samples requirement
            if y is not None and len(X) < self.n_neighbors:
                raise ValueError(f"Need at least {self.n_neighbors} samples, got {len(X)}")
            
            # Scale features if enabled
            if self.auto_scale and self.scaler is not None:
                if y is not None:
                    # Training: fit and transform
                    X_scaled = self.scaler.fit_transform(X)
                else:
                    # Prediction: transform only
                    if not hasattr(self.scaler, 'scale_'):
                        raise ValueError("Scaler not fitted. Call fit() first.")
                    X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {str(e)}")
            raise
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the KNN model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training targets
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metadata
        """
        try:
            start_time = time.time()
            logger.info(f"🚀 Training KNN with n_neighbors={self.n_neighbors}")
            
            # Create model and scaler
            self._create_model()
            self._create_scaler()
            
            # Validate and prepare data
            X_processed, y_processed = self._validate_and_prepare_data(X, y)
            
            # Store training data for analysis
            self.X_train_ = X.copy()
            self.y_train_ = y.copy()
            self.X_train_scaled_ = X_processed.copy()
            
            # Train the model
            self.model.fit(X_processed, y_processed)
            self.is_fitted = True
            
            # Record training time
            self.training_time_ = time.time() - start_time
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_processed)
            train_r2 = r2_score(y_processed, train_predictions)
            train_mse = mean_squared_error(y_processed, train_predictions)
            
            # Perform quick cross-validation
            cv_scores = cross_val_score(
                self.model, X_processed, y_processed,
                cv=min(5, len(X) // 2),
                scoring='r2',
                n_jobs=self.config.get('n_jobs', 1)
            )
            
            # Clear analysis cache
            self._analysis_cache.clear()
            
            training_result = {
                'model_fitted': True,
                'training_time': self.training_time_,
                'training_score': train_r2,
                'training_mse': train_mse,
                'cv_mean_score': cv_scores.mean(),
                'cv_std_score': cv_scores.std(),
                'effective_n_neighbors': min(self.n_neighbors, len(X)),
                'data_shape': X.shape,
                'scaling_applied': self.auto_scale,
                'algorithm_used': self.algorithm,
                'metric_used': self.metric
            }
            
            logger.info(f"✅ Training completed in {self.training_time_:.3f}s, CV R² = {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            return training_result
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                'model_fitted': False,
                'error': error_msg,
                'training_time': getattr(self, 'training_time_', 0)
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            start_time = time.time()
            
            # Prepare data
            X_processed, _ = self._validate_and_prepare_data(X)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            
            # Record prediction time
            prediction_time = time.time() - start_time
            self.prediction_times_.append(prediction_time)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            raise
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score on test data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features
        y : np.ndarray of shape (n_samples,)
            Test targets
            
        Returns:
        --------
        float
            R² score
        """
        try:
            if not self.is_fitted:
                logger.warning("⚠️ Model not fitted, returning 0 score")
                return 0.0
            
            predictions = self.predict(X)
            return r2_score(y, predictions)
            
        except Exception as e:
            logger.error(f"❌ Scoring failed: {str(e)}")
            return 0.0
    
    def get_neighbors_info(self, X: np.ndarray, return_distance: bool = True) -> Dict[str, Any]:
        """
        Get information about neighbors for given samples.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Query samples
        return_distance : bool, default=True
            Whether to return distances to neighbors
            
        Returns:
        --------
        Dict[str, Any]
            Neighbor information including indices and optionally distances
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before getting neighbors")
            
            # Prepare data
            X_processed, _ = self._validate_and_prepare_data(X)
            
            # Get neighbors
            if return_distance:
                distances, indices = self.model.kneighbors(
                    X_processed, return_distance=True
                )
                return {
                    'neighbor_indices': indices,
                    'neighbor_distances': distances,
                    'mean_distances': distances.mean(axis=1),
                    'min_distances': distances.min(axis=1),
                    'max_distances': distances.max(axis=1),
                    'distance_std': distances.std(axis=1)
                }
            else:
                indices = self.model.kneighbors(
                    X_processed, return_distance=False
                )
                return {
                    'neighbor_indices': indices
                }
                
        except Exception as e:
            logger.error(f"❌ Getting neighbors info failed: {str(e)}")
            return {'error': str(e)}
    
    def analyze_optimal_k(self, X: np.ndarray, y: np.ndarray, 
                         k_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Analyze optimal K value using cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training targets
        k_range : Tuple[int, int], optional
            Range of K values to test. If None, uses config value
            
        Returns:
        --------
        Dict[str, Any]
            K optimization analysis results
        """
        try:
            # Use provided range or default from config
            if k_range is None:
                k_range = self.config.get('k_range', (1, 20))
            
            k_min, k_max = k_range
            k_max = min(k_max, len(X) - 1)  # Can't have more neighbors than samples
            
            if k_max <= k_min:
                k_max = k_min + 5
            
            k_values = list(range(k_min, k_max + 1))
            
            logger.info(f"🔍 Analyzing optimal K in range {k_min}-{k_max}")
            
            # Prepare data
            X_processed, y_processed = self._validate_and_prepare_data(X, y)
            
            # Test different K values
            train_scores = []
            validation_scores = []
            cv_scores_mean = []
            cv_scores_std = []
            
            original_k = self.n_neighbors
            
            for k in k_values:
                # Create temporary model with current K
                temp_model = KNeighborsRegressor(
                    n_neighbors=k,
                    weights=self.weights,
                    algorithm=self.algorithm,
                    metric=self.metric,
                    p=self.p if self.metric == 'minkowski' else 2,
                    n_jobs=self.config.get('n_jobs', 1)
                )
                
                # Train and get training score
                temp_model.fit(X_processed, y_processed)
                train_pred = temp_model.predict(X_processed)
                train_score = r2_score(y_processed, train_pred)
                train_scores.append(train_score)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    temp_model, X_processed, y_processed,
                    cv=min(5, len(X) // 3),
                    scoring='r2',
                    n_jobs=self.config.get('n_jobs', 1)
                )
                
                cv_scores_mean.append(cv_scores.mean())
                cv_scores_std.append(cv_scores.std())
            
            # Restore original K
            self.n_neighbors = original_k
            
            # Find optimal K
            cv_scores_array = np.array(cv_scores_mean)
            optimal_idx = np.argmax(cv_scores_array)
            optimal_k = k_values[optimal_idx]
            optimal_score = cv_scores_array[optimal_idx]
            
            # Calculate improvement potential
            current_k_idx = k_values.index(original_k) if original_k in k_values else 0
            current_score = cv_scores_array[current_k_idx] if original_k in k_values else cv_scores_array[0]
            improvement = optimal_score - current_score
            
            result = {
                'k_values': k_values,
                'train_scores': train_scores,
                'cv_scores_mean': cv_scores_mean,
                'cv_scores_std': cv_scores_std,
                'optimal_k': optimal_k,
                'optimal_score': optimal_score,
                'current_k': original_k,
                'current_score': current_score,
                'improvement_potential': improvement,
                'k_range_tested': (k_min, k_max),
                'recommendation': {
                    'suggested_k': optimal_k,
                    'confidence': 'high' if improvement > 0.05 else 'medium' if improvement > 0.02 else 'low',
                    'reason': f"K={optimal_k} shows {improvement:.3f} improvement in CV R²"
                }
            }
            
            # Cache result
            self._analysis_cache['optimal_k_analysis'] = result
            
            logger.info(f"✅ Optimal K analysis completed: K={optimal_k} (score={optimal_score:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Optimal K analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def analyze_distance_metrics(self, X: np.ndarray, y: np.ndarray, 
                                metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare performance across different distance metrics.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training targets
        metrics : List[str], optional
            List of metrics to test
            
        Returns:
        --------
        Dict[str, Any]
            Distance metric comparison results
        """
        try:
            if metrics is None:
                metrics = ['euclidean', 'manhattan', 'chebyshev']
                # Add minkowski if not already included
                if self.metric == 'minkowski' and 'minkowski' not in metrics:
                    metrics.append('minkowski')
            
            logger.info(f"🔍 Analyzing distance metrics: {metrics}")
            
            # Prepare data
            X_processed, y_processed = self._validate_and_prepare_data(X, y)
            
            results = {}
            original_metric = self.metric
            original_p = self.p
            
            for metric in metrics:
                try:
                    # Create model with current metric
                    model_params = {
                        'n_neighbors': self.n_neighbors,
                        'weights': self.weights,
                        'algorithm': self.algorithm,
                        'metric': metric,
                        'n_jobs': self.config.get('n_jobs', 1)
                    }
                    
                    if metric == 'minkowski':
                        model_params['p'] = self.p
                    
                    temp_model = KNeighborsRegressor(**model_params)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        temp_model, X_processed, y_processed,
                        cv=min(5, len(X) // 3),
                        scoring='r2',
                        n_jobs=self.config.get('n_jobs', 1)
                    )
                    
                    # Training score
                    temp_model.fit(X_processed, y_processed)
                    train_pred = temp_model.predict(X_processed)
                    train_score = r2_score(y_processed, train_pred)
                    
                    results[metric] = {
                        'cv_score_mean': cv_scores.mean(),
                        'cv_score_std': cv_scores.std(),
                        'train_score': train_score,
                        'cv_scores': cv_scores.tolist()
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to test metric {metric}: {str(e)}")
                    results[metric] = {'error': str(e)}
            
            # Restore original settings
            self.metric = original_metric
            self.p = original_p
            
            # Find best metric
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                best_metric = max(valid_results.keys(), 
                                key=lambda k: valid_results[k]['cv_score_mean'])
                best_score = valid_results[best_metric]['cv_score_mean']
                current_score = valid_results.get(original_metric, {}).get('cv_score_mean', 0)
                improvement = best_score - current_score if current_score > 0 else 0
                
                analysis_result = {
                    'metric_results': results,
                    'best_metric': best_metric,
                    'best_score': best_score,
                    'current_metric': original_metric,
                    'current_score': current_score,
                    'improvement_potential': improvement,
                    'recommendation': {
                        'suggested_metric': best_metric,
                        'confidence': 'high' if improvement > 0.03 else 'medium' if improvement > 0.01 else 'low',
                        'reason': f"Metric '{best_metric}' shows {improvement:.3f} improvement"
                    }
                }
            else:
                analysis_result = {
                    'metric_results': results,
                    'error': 'No valid metric results obtained'
                }
            
            # Cache result
            self._analysis_cache['distance_metrics_analysis'] = analysis_result
            
            logger.info("✅ Distance metrics analysis completed")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Distance metrics analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def analyze_algorithm_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze performance of different neighbor search algorithms.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training targets
            
        Returns:
        --------
        Dict[str, Any]
            Algorithm performance analysis results
        """
        try:
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            
            logger.info(f"🔍 Analyzing algorithm performance: {algorithms}")
            
            # Prepare data
            X_processed, y_processed = self._validate_and_prepare_data(X, y)
            
            results = {}
            original_algorithm = self.algorithm
            
            for algo in algorithms:
                try:
                    # Create model with current algorithm
                    temp_model = KNeighborsRegressor(
                        n_neighbors=self.n_neighbors,
                        weights=self.weights,
                        algorithm=algo,
                        metric=self.metric,
                        p=self.p if self.metric == 'minkowski' else 2,
                        n_jobs=self.config.get('n_jobs', 1)
                    )
                    
                    # Measure training time
                    start_time = time.time()
                    temp_model.fit(X_processed, y_processed)
                    training_time = time.time() - start_time
                    
                    # Measure prediction time
                    start_time = time.time()
                    predictions = temp_model.predict(X_processed[:min(100, len(X_processed))])
                    prediction_time = time.time() - start_time
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(
                        temp_model, X_processed, y_processed,
                        cv=min(3, len(X) // 3),  # Smaller CV for speed
                        scoring='r2'
                    )
                    
                    results[algo] = {
                        'training_time': training_time,
                        'prediction_time': prediction_time,
                        'cv_score_mean': cv_scores.mean(),
                        'cv_score_std': cv_scores.std(),
                        'samples_per_second': len(X_processed) / training_time if training_time > 0 else float('inf')
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to test algorithm {algo}: {str(e)}")
                    results[algo] = {'error': str(e)}
            
            # Restore original algorithm
            self.algorithm = original_algorithm
            
            # Find fastest algorithm
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                fastest_training = min(valid_results.keys(), 
                                     key=lambda k: valid_results[k]['training_time'])
                fastest_prediction = min(valid_results.keys(), 
                                       key=lambda k: valid_results[k]['prediction_time'])
                
                analysis_result = {
                    'algorithm_results': results,
                    'fastest_training': fastest_training,
                    'fastest_prediction': fastest_prediction,
                    'current_algorithm': original_algorithm,
                    'recommendation': {
                        'for_training': fastest_training,
                        'for_prediction': fastest_prediction,
                        'overall': 'auto'  # Generally the best choice
                    }
                }
            else:
                analysis_result = {
                    'algorithm_results': results,
                    'error': 'No valid algorithm results obtained'
                }
            
            # Cache result
            self._analysis_cache['algorithm_performance_analysis'] = analysis_result
            
            logger.info("✅ Algorithm performance analysis completed")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Algorithm performance analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            if not self.is_fitted:
                return {'error': 'Model must be fitted first'}
            
            summary = {
                'model_parameters': {
                    'n_neighbors': self.n_neighbors,
                    'weights': self.weights,
                    'algorithm': self.algorithm,
                    'metric': self.metric,
                    'p': self.p,
                    'auto_scale': self.auto_scale
                },
                'training_info': {
                    'training_time': self.training_time_,
                    'data_shape': self.X_train_.shape if self.X_train_ is not None else None,
                    'effective_neighbors': min(self.n_neighbors, len(self.X_train_)) if self.X_train_ is not None else None
                },
                'prediction_performance': {
                    'mean_prediction_time': np.mean(self.prediction_times_) if self.prediction_times_ else None,
                    'total_predictions': len(self.prediction_times_)
                },
                'cached_analyses': list(self._analysis_cache.keys())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Performance summary failed: {str(e)}")
            return {'error': str(e)}
    
    def get_hyperparameter_config(self) -> Dict[str, Any]:
        """Get current hyperparameter configuration."""
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'p': self.p,
            'auto_scale': self.auto_scale,
            'random_state': self.random_state
        }
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._analysis_cache.clear()
        logger.info("🗑️ Analysis cache cleared")
    
    def __repr__(self) -> str:
        """String representation of the KNN Core."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"KNNCore(n_neighbors={self.n_neighbors}, "
                f"weights='{self.weights}', metric='{self.metric}', "
                f"algorithm='{self.algorithm}', status={status})")

# Utility functions for KNN-specific operations
def calculate_optimal_k_rule(n_samples: int) -> int:
    """Calculate optimal K using rule of thumb."""
    return max(1, int(np.sqrt(n_samples)))


def calculate_curse_dimensionality_factor(n_features: int, n_samples: int) -> float:
    """Calculate curse of dimensionality factor."""
    if n_samples == 0:
        return float('inf')
    return n_features / np.log(n_samples)


def estimate_memory_usage(n_samples: int, n_features: int, algorithm: str) -> Dict[str, float]:
    """Estimate memory usage for different algorithms."""
    base_memory = n_samples * n_features * 8  # 8 bytes per float64
    
    if algorithm == 'brute':
        # Brute force stores distance matrix
        memory_mb = (base_memory + n_samples * n_samples * 8) / (1024 * 1024)
    elif algorithm in ['ball_tree', 'kd_tree']:
        # Tree algorithms have logarithmic overhead
        memory_mb = base_memory * (1 + np.log(n_samples)) / (1024 * 1024)
    else:
        # Auto algorithm
        memory_mb = base_memory * 1.2 / (1024 * 1024)
    
    return {
        'estimated_memory_mb': memory_mb,
        'base_data_mb': base_memory / (1024 * 1024),
        'overhead_factor': memory_mb / (base_memory / (1024 * 1024)) if base_memory > 0 else 1
    }