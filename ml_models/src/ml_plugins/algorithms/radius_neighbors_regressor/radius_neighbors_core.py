"""
Radius Neighbors Regressor - Core Implementation
=============================================

This module contains the core functionality for the Radius Neighbors Regressor algorithm,
including model training, prediction, and fundamental operations.

Author: Bachelor Thesis Project
Date: June 2025
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, Any, Optional, Tuple, Union, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class RadiusNeighborsCore(BaseEstimator, RegressorMixin):
    """
    Core implementation of Radius Neighbors Regressor algorithm.
    
    This class handles the fundamental operations including model training,
    prediction, data preprocessing, and basic algorithm configuration.
    """
    
    def __init__(self, 
                radius: float = 1.0,
                weights: str = 'uniform',
                algorithm: str = 'auto',
                leaf_size: int = 30,
                metric: str = 'minkowski',
                p: int = 2,
                metric_params: Optional[Dict] = None,
                n_jobs: Optional[int] = None,
                auto_scale: bool = True,
                adaptive_radius: bool = False,
                scaler_type: str = 'standard',
                random_state: Optional[int] = None):
        """
        Initialize the Radius Neighbors Core component.
        
        Parameters:
        -----------
        radius : float, default=1.0
            Range of parameter space to use for neighbor search
        weights : str, default='uniform'
            Weight function for predictions ('uniform' or 'distance')
        algorithm : str, default='auto'
            Algorithm to use for neighbor search
        leaf_size : int, default=30
            Leaf size for tree algorithms
        metric : str, default='minkowski'
            Distance metric for neighbor search
        p : int, default=2
            Parameter for Minkowski metric
        metric_params : dict, optional
            Additional parameters for distance metric
        n_jobs : int, optional
            Number of parallel jobs
        auto_scale : bool, default=True
            Whether to automatically scale features
        adaptive_radius : bool, default=False
            Whether to use adaptive radius based on data density
        scaler_type : str, default='standard'
            Type of scaler to use ('standard', 'minmax', 'robust')
        random_state : int, optional
            Random state for reproducibility (used for data splitting in analysis)
        """
        # Core algorithm parameters
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.random_state = random_state  # Add this line
        
        # Advanced parameters
        self.auto_scale = auto_scale
        self.adaptive_radius = adaptive_radius
        self.scaler_type = scaler_type
        
        # Model state variables
        self.model_ = None
        self.scaler_ = None
        self.is_fitted_ = False
        self.effective_radius_ = radius
        self.training_time_ = 0.0
        
        # Data storage
        self.X_train_ = None
        self.X_train_scaled_ = None
        self.y_train_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.n_samples_in_ = None
        
        # Configuration
        self._min_samples_required = 5
        
    def get_name(self) -> str:
        """Get the algorithm name."""
        return "Radius Neighbors Regressor"
    
    def get_description(self) -> str:
        """Get the algorithm description."""
        return ("A regression algorithm that makes predictions based on neighbors "
                "within a specified radius, using local density information.")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'RadiusNeighborsCore':
        """
        Fit the Radius Neighbors Regressor model.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : pd.Series or np.ndarray
            Training targets
            
        Returns:
        --------
        self : RadiusNeighborsCore
            Fitted estimator
        """
        start_time = time.time()
        
        try:
            # Convert inputs to numpy arrays and store original data
            self.X_train_ = self._convert_to_array(X)
            self.y_train_ = self._convert_to_array(y)
            
            # Store data characteristics
            self.n_samples_in_, self.n_features_in_ = self.X_train_.shape
            self.feature_names_ = self._extract_feature_names(X)
            
            # Validate inputs
            self._validate_inputs()
            
            # Create and apply scaler if needed
            if self.auto_scale:
                self.scaler_ = self._create_scaler()
                self.X_train_scaled_ = self.scaler_.fit_transform(self.X_train_)
            else:
                self.X_train_scaled_ = self.X_train_.copy()
                self.scaler_ = None
            
            # Calculate effective radius
            self.effective_radius_ = self._calculate_effective_radius()
            
            # Create and fit the model
            self.model_ = self._create_model_instance()
            self.model_.fit(self.X_train_scaled_, self.y_train_)
            
            # Update state
            self.is_fitted_ = True
            self.training_time_ = time.time() - start_time
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit Radius Neighbors model: {str(e)}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Test features
            
        Returns:
        --------
        np.ndarray
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        try:
            # Convert to array
            X_array = self._convert_to_array(X)
            
            # Validate input shape
            if X_array.shape[1] != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {X_array.shape[1]}")
            
            # Apply scaling if used during training
            if self.scaler_ is not None:
                X_scaled = self.scaler_.transform(X_array)
            else:
                X_scaled = X_array
            
            # Make predictions
            return self.model_.predict(X_scaled)
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def train_model(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Train the model and return comprehensive training information.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Training features
        y : pd.Series or np.ndarray
            Training targets
            
        Returns:
        --------
        Dict[str, Any]
            Training results and model information
        """
        try:
            # Fit the model
            self.fit(X, y)
            
            # Calculate training metrics
            train_predictions = self.predict(self.X_train_)
            train_r2 = r2_score(self.y_train_, train_predictions)
            train_mse = mean_squared_error(self.y_train_, train_predictions)
            
            # Analyze neighborhoods
            neighborhood_info = self._analyze_training_neighborhoods()
            
            # Prepare results
            training_results = {
                'model_fitted': True,
                'training_time': self.training_time_,
                'n_samples': self.n_samples_in_,
                'n_features': self.n_features_in_,
                'effective_radius': self.effective_radius_,
                'original_radius': self.radius,
                'scaling_applied': self.scaler_ is not None,
                'scaler_type': self.scaler_type if self.scaler_ is not None else None,
                'train_r2_score': train_r2,
                'train_mse': train_mse,
                'neighborhood_analysis': neighborhood_info,
                'algorithm_config': {
                    'algorithm': self.algorithm,
                    'weights': self.weights,
                    'metric': self.metric,
                    'p': self.p,
                    'leaf_size': self.leaf_size
                }
            }
            
            return training_results
            
        except Exception as e:
            return {
                'model_fitted': False,
                'error': str(e),
                'training_time': 0.0
            }
    
    def predict_with_neighborhood_info(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make predictions with detailed neighborhood information.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Test features
            
        Returns:
        --------
        Dict[str, Any]
            Predictions with neighborhood analysis
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet.")
        
        try:
            # Convert and scale input
            X_array = self._convert_to_array(X)
            if self.scaler_ is not None:
                X_scaled = self.scaler_.transform(X_array)
            else:
                X_scaled = X_array
            
            # Make predictions
            predictions = self.model_.predict(X_scaled)
            
            # Analyze neighborhoods for predictions
            neighborhood_analysis = self._analyze_prediction_neighborhoods(X_scaled)
            
            # Calculate prediction confidence
            confidence_scores = self._calculate_prediction_confidence(X_scaled, neighborhood_analysis)
            
            # Calculate outlier scores
            outlier_scores = self._calculate_outlier_scores(X_scaled, neighborhood_analysis)
            
            return {
                'predictions': predictions,
                'neighborhood_analysis': neighborhood_analysis,
                'confidence_scores': confidence_scores,
                'outlier_scores': outlier_scores,
                'n_predictions': len(predictions),
                'effective_radius': self.effective_radius_
            }
            
        except Exception as e:
            return {
                'predictions': None,
                'error': str(e)
            }
    
# Update the get_hyperparameter_config method

    def get_hyperparameter_config(self) -> Dict[str, Any]:
        """
        Get hyperparameter configuration for the algorithm.
        
        Returns:
        --------
        Dict[str, Any]
            Hyperparameter configuration
        """
        return {
            'radius': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'description': 'Radius for neighbor search'
            },
            'weights': {
                'type': 'select',
                'options': ['uniform', 'distance'],
                'default': 'uniform',
                'description': 'Weight function for predictions'
            },
            'algorithm': {
                'type': 'select',
                'options': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'default': 'auto',
                'description': 'Algorithm for neighbor search'
            },
            'metric': {
                'type': 'select',
                'options': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
                'default': 'minkowski',
                'description': 'Distance metric'
            },
            'p': {
                'type': 'int',
                'default': 2,
                'min': 1,
                'max': 10,
                'description': 'Parameter for Minkowski metric'
            },
            'auto_scale': {
                'type': 'boolean',
                'default': True,
                'description': 'Automatically scale features'
            },
            'adaptive_radius': {
                'type': 'boolean',
                'default': False,
                'description': 'Use adaptive radius based on data density'
            },
            'scaler_type': {
                'type': 'select',
                'options': ['standard', 'minmax', 'robust'],
                'default': 'standard',
                'description': 'Type of feature scaler'
            },
            'leaf_size': {
                'type': 'int',
                'default': 30,
                'min': 1,
                'max': 100,
                'description': 'Leaf size for tree algorithms'
            },
            'random_state': {
                'type': 'int',
                'default': None,
                'min': 0,
                'max': 9999,
                'description': 'Random state for reproducibility'
            }
        }
    
    def create_model_instance(self) -> RadiusNeighborsRegressor:
        """
        Create a new instance of the underlying sklearn model.
        
        Returns:
        --------
        RadiusNeighborsRegressor
            Configured sklearn model instance
        """
        return self._create_model_instance()
    
    # ==================== PRIVATE METHODS ====================
    
    def _convert_to_array(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """Convert various data types to numpy array."""
        if hasattr(data, 'values'):
            return data.values
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
    
    def _extract_feature_names(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """Extract feature names from input data."""
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        else:
            return [f'feature_{i}' for i in range(X.shape[1])]
    
    def _validate_inputs(self) -> None:
        """Validate input data."""
        if self.n_samples_in_ < self._min_samples_required:
            raise ValueError(f"Need at least {self._min_samples_required} samples, got {self.n_samples_in_}")
        
        if self.n_features_in_ < 1:
            raise ValueError("Need at least 1 feature")
        
        if len(self.y_train_) != self.n_samples_in_:
            raise ValueError("Number of samples in X and y must be equal")
        
        if not np.isfinite(self.X_train_).all():
            raise ValueError("X contains NaN or infinite values")
        
        if not np.isfinite(self.y_train_).all():
            raise ValueError("y contains NaN or infinite values")
    
    def _create_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        """Create the appropriate scaler based on configuration."""
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if self.scaler_type not in scaler_map:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        return scaler_map[self.scaler_type]
    
    def _calculate_effective_radius(self) -> float:
        """Calculate the effective radius to use for the model."""
        if not self.adaptive_radius:
            return self.radius
        
        try:
            # Calculate adaptive radius based on data density
            n_samples = self.X_train_scaled_.shape[0]
            n_features = self.X_train_scaled_.shape[1]
            
            # Estimate data volume and density
            data_range = np.ptp(self.X_train_scaled_, axis=0).mean()
            volume_estimate = data_range ** n_features
            density_estimate = n_samples / volume_estimate if volume_estimate > 0 else 1.0
            
            # Adaptive radius calculation
            base_radius = self.radius
            density_factor = min(2.0, max(0.5, 1.0 / np.sqrt(density_estimate)))
            adaptive_radius = base_radius * density_factor
            
            return float(adaptive_radius)
            
        except Exception:
            # Fallback to original radius if adaptive calculation fails
            return self.radius
    
    def _create_model_instance(self) -> RadiusNeighborsRegressor:
        """Create the sklearn RadiusNeighborsRegressor instance."""
        return RadiusNeighborsRegressor(
            radius=self.effective_radius_,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )
    
    def _analyze_training_neighborhoods(self) -> Dict[str, Any]:
        """Analyze neighborhood characteristics of training data."""
        try:
            # Get neighbor information
            neighbor_indices = self.model_.radius_neighbors(self.X_train_scaled_, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            
            # Calculate statistics
            coverage = (np.sum(neighbor_counts > 0) / len(neighbor_counts)) * 100
            mean_neighbors = np.mean(neighbor_counts)
            std_neighbors = np.std(neighbor_counts)
            isolated_points = np.sum(neighbor_counts == 1)  # Only themselves
            
            return {
                'mean_neighbors': float(mean_neighbors),
                'std_neighbors': float(std_neighbors),
                'coverage_percentage': float(coverage),
                'isolated_points': int(isolated_points),
                'min_neighbors': int(np.min(neighbor_counts)),
                'max_neighbors': int(np.max(neighbor_counts)),
                'median_neighbors': float(np.median(neighbor_counts))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_prediction_neighborhoods(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Analyze neighborhoods for prediction points."""
        try:
            neighbor_indices = self.model_.radius_neighbors(X_scaled, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            
            return {
                'neighbor_counts': neighbor_counts.tolist(),
                'mean_neighbors': float(np.mean(neighbor_counts)),
                'coverage_points': int(np.sum(neighbor_counts > 0)),
                'isolated_predictions': int(np.sum(neighbor_counts == 0))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_prediction_confidence(self, X_scaled: np.ndarray, neighborhood_info: Dict[str, Any]) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        try:
            neighbor_counts = np.array(neighborhood_info.get('neighbor_counts', []))
            
            if len(neighbor_counts) == 0:
                return np.zeros(len(X_scaled))
            
            # Confidence based on number of neighbors
            max_neighbors = max(neighbor_counts) if max(neighbor_counts) > 0 else 1
            confidence = neighbor_counts / max_neighbors
            
            return confidence
            
        except Exception:
            return np.zeros(len(X_scaled))
    
    def _calculate_outlier_scores(self, X_scaled: np.ndarray, neighborhood_info: Dict[str, Any]) -> np.ndarray:
        """Calculate outlier scores for prediction points."""
        try:
            neighbor_counts = np.array(neighborhood_info.get('neighbor_counts', []))
            
            if len(neighbor_counts) == 0:
                return np.zeros(len(X_scaled))
            
            # Simple outlier score: inverse of neighbor density
            outlier_scores = 1.0 / (neighbor_counts + 1)  # +1 to avoid division by zero
            
            return outlier_scores
            
        except Exception:
            return np.zeros(len(X_scaled))
    
    def is_compatible_with_data(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[bool, str]:
        """
        Check if the algorithm is compatible with the given data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature data
        y : pd.Series or np.ndarray, optional
            Target data
            
        Returns:
        --------
        Tuple[bool, str]
            (is_compatible, message)
        """
        try:
            X_array = self._convert_to_array(X)
            n_samples, n_features = X_array.shape
            
            if n_samples < self._min_samples_required:
                return False, f"Requires at least {self._min_samples_required} samples, got {n_samples}"
            
            if y is not None:
                y_array = self._convert_to_array(y)
                if not np.issubdtype(y_array.dtype, np.number):
                    return False, "Requires numerical target values for regression"
                
                if len(y_array) != n_samples:
                    return False, "Number of samples in X and y must be equal"
            
            return True, f"Compatible with {n_samples} samples and {n_features} features"
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
# Update the get_model_params method

    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters and state."""
        return {
            'algorithm': 'Radius Neighbors Regressor',
            'radius': self.effective_radius_ if hasattr(self, 'effective_radius_') else self.radius,
            'original_radius': self.radius,
            'weights': self.weights,
            'metric': self.metric,
            'algorithm_type': self.algorithm,
            'p': self.p,
            'leaf_size': self.leaf_size,
            'auto_scale': self.auto_scale,
            'adaptive_radius': self.adaptive_radius,
            'scaler_type': self.scaler_type,
            'random_state': self.random_state,  # Add this line
            'n_features': self.n_features_in_ if hasattr(self, 'n_features_in_') else None,
            'n_samples': self.n_samples_in_ if hasattr(self, 'n_samples_in_') else None,
            'is_fitted': self.is_fitted_,
            'training_time': self.training_time_ if hasattr(self, 'training_time_') else 0.0
        }


# ==================== TESTING FUNCTIONS ====================

def test_core_functionality():
    """Test the core functionality of RadiusNeighborsCore."""
    print("🧪 Testing Radius Neighbors Core Functionality...")
    
    try:
        # Generate test data
        np.random.seed(42)
        n_samples, n_features = 100, 4
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(n_samples)  # Linear relationship with noise
        
        # Test 1: Basic initialization
        print("✅ Test 1: Initialization")
        core = RadiusNeighborsCore(radius=1.0, auto_scale=True)
        assert core.radius == 1.0
        assert core.auto_scale == True
        assert not core.is_fitted_
        
        # Test 2: Model fitting
        print("✅ Test 2: Model fitting")
        training_results = core.train_model(X, y)
        assert training_results['model_fitted'] == True
        assert core.is_fitted_ == True
        assert 'train_r2_score' in training_results
        
        # Test 3: Predictions
        print("✅ Test 3: Basic predictions")
        predictions = core.predict(X[:10])
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
        
        # Test 4: Detailed predictions
        print("✅ Test 4: Detailed predictions")
        detailed_results = core.predict_with_neighborhood_info(X[:5])
        assert 'predictions' in detailed_results
        assert 'neighborhood_analysis' in detailed_results
        assert len(detailed_results['predictions']) == 5
        
        # Test 5: Hyperparameter configuration
        print("✅ Test 5: Hyperparameter configuration")
        config = core.get_hyperparameter_config()
        assert 'radius' in config
        assert 'weights' in config
        assert 'algorithm' in config
        
        # Test 6: Compatibility check
        print("✅ Test 6: Data compatibility")
        is_compatible, message = core.is_compatible_with_data(X, y)
        assert is_compatible == True
        
        # Test 7: Model parameters
        print("✅ Test 7: Model parameters")
        params = core.get_model_params()
        assert params['is_fitted'] == True
        assert 'n_features' in params
        assert 'n_samples' in params
        
        print("🎉 All core functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_core_functionality()