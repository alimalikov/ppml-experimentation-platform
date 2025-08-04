"""
Radius Neighbors Regressor - Analysis Implementation
=================================================

This module contains comprehensive analysis functionality for the Radius Neighbors Regressor,
including performance profiling, feature importance, cross-validation, and comparative analysis.

Author: Bachelor Thesis Project
Date: June 2025
"""

import numpy as np
import pandas as pd
import time
import psutil
import platform
import traceback
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from .radius_neighbors_core import RadiusNeighborsCore

warnings.filterwarnings('ignore')

class RadiusNeighborsAnalysis:
    """
    Comprehensive analysis component for Radius Neighbors Regressor.
    
    This class provides various analytical methods including performance profiling,
    feature importance analysis, cross-validation, and comparative studies.
    """
    
    def __init__(self, core: RadiusNeighborsCore):
        """
        Initialize the analysis component.
        
        Parameters:
        -----------
        core : RadiusNeighborsCore
            The core component containing the fitted model and data
        """
        self.core = core
        self.analysis_cache_ = {}
        
    def analyze_radius_behavior(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of radius behavior and its impact on model performance.
        
        Returns:
        --------
        Dict[str, Any]
            Complete radius behavior analysis
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before analysis'}
        
        try:
            analysis_results = {
                'radius_coverage': self._analyze_radius_coverage(),
                'density_distribution': self._analyze_density_distribution(),
                'neighborhood_characteristics': self._analyze_neighborhood_characteristics(),
                'distance_metrics': self._analyze_distance_metrics(),
                'optimal_radius': self._analyze_optimal_radius(),
                'outlier_behavior': self._analyze_outlier_behavior()
            }
            
            self.analysis_cache_['radius_behavior'] = analysis_results
            return analysis_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_cross_validation(self, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation analysis.
        
        Parameters:
        -----------
        cv_folds : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, Any]
            Cross-validation results and analysis
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before analysis'}
        
        try:
            # Prepare cross-validation
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Create model for CV
            cv_model = RadiusNeighborsRegressor(
                radius=self.core.effective_radius_,
                weights=self.core.weights,
                algorithm=self.core.algorithm,
                metric=self.core.metric,
                p=self.core.p
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                cv_model, self.core.X_train_scaled_, self.core.y_train_, 
                cv=cv, scoring='r2'
            )
            
            # Detailed fold analysis
            fold_results = []
            for fold, (train_idx, val_idx) in enumerate(cv.split(self.core.X_train_scaled_)):
                fold_analysis = self._analyze_cv_fold(fold, train_idx, val_idx)
                fold_results.append(fold_analysis)
            
            # Adaptive behavior analysis
            adaptive_analysis = self._analyze_adaptive_behavior()
            
            return {
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': float(np.mean(cv_scores)),
                'std_cv_score': float(np.std(cv_scores)),
                'cv_score_range': float(np.max(cv_scores) - np.min(cv_scores)),
                'fold_details': fold_results,
                'adaptive_behavior': adaptive_analysis,
                'stability_metrics': {
                    'coefficient_of_variation': float(np.std(cv_scores) / np.mean(cv_scores)) if np.mean(cv_scores) != 0 else 0,
                    'consistency_score': 1.0 - (np.std(cv_scores) / np.mean(cv_scores)) if np.mean(cv_scores) != 0 else 0
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze feature importance using permutation-based method.
        
        Returns:
        --------
        Dict[str, Any]
            Feature importance analysis results
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before analysis'}
        
        try:
            # Get baseline performance
            baseline_performance = self._get_baseline_performance()
            
            # Permutation-based feature importance
            importance_scores = {}
            importance_changes = {}
            
            for i, feature_name in enumerate(self.core.feature_names_):
                # Permute feature and measure performance change
                permuted_scores = []
                for _ in range(5):  # Multiple permutations for stability
                    permuted_performance = self._analyze_feature_permutation(i)
                    if 'r2' in permuted_performance:
                        score_change = baseline_performance['r2'] - permuted_performance['r2']
                        permuted_scores.append(score_change)
                
                if permuted_scores:
                    importance_scores[feature_name] = np.mean(permuted_scores)
                    importance_changes[feature_name] = {
                        'mean_change': np.mean(permuted_scores),
                        'std_change': np.std(permuted_scores),
                        'max_change': np.max(permuted_scores)
                    }
                else:
                    importance_scores[feature_name] = 0.0
                    importance_changes[feature_name] = {'mean_change': 0.0, 'std_change': 0.0, 'max_change': 0.0}
            
            # Feature interaction analysis
            interaction_analysis = self._analyze_feature_interactions()
            
            # Dimensionality impact
            dimensionality_analysis = self._analyze_dimensionality_impact()
            
            return {
                'baseline_performance': baseline_performance,
                'feature_importance_scores': importance_scores,
                'importance_changes': importance_changes,
                'feature_ranking': sorted(importance_scores.items(), key=lambda x: x[1], reverse=True),
                'interaction_analysis': interaction_analysis,
                'dimensionality_impact': dimensionality_analysis,
                'top_features': self._get_top_features(importance_scores, 3)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def profile_performance(self) -> Dict[str, Any]:
        """
        Comprehensive performance profiling of the algorithm.
        
        Returns:
        --------
        Dict[str, Any]
            Complete performance profile
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before profiling'}
        
        try:
            # System information
            system_info = self._get_system_info()
            
            # Data characteristics profiling
            data_characteristics = self._profile_data_characteristics()
            
            # Operation timing analysis
            operations_timing = self._profile_operations_timing()
            
            # Memory usage analysis
            memory_usage = self._profile_memory_usage()
            
            # Scalability analysis
            scalability_analysis = self._analyze_scalability()
            
            # Algorithm comparison timing
            algorithm_timing = self._profile_algorithm_timing()
            
            # Efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(operations_timing, memory_usage)
            
            # Performance recommendations
            recommendations = self._generate_performance_recommendations(
                operations_timing, efficiency_metrics, algorithm_timing
            )
            
            # Bottleneck analysis
            bottleneck_analysis = self._analyze_bottlenecks(operations_timing, efficiency_metrics)
            
            performance_profile = {
                'system_info': system_info,
                'data_characteristics': data_characteristics,
                'operations_timing': operations_timing,
                'memory_usage_mb': memory_usage,
                'scalability_analysis': scalability_analysis,
                'algorithm_timing': algorithm_timing,
                'efficiency_metrics': efficiency_metrics,
                'performance_recommendations': recommendations,
                'bottleneck_analysis': bottleneck_analysis
            }
            
            self.analysis_cache_['performance_profile'] = performance_profile
            return performance_profile
            
        except Exception as e:
            return {'error': str(e)}
    
    def compare_with_knn(self) -> Dict[str, Any]:
        """
        Compare RadiusNeighbors performance with standard KNN.
        
        Returns:
        --------
        Dict[str, Any]
            Comparison results with KNN algorithms
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before comparison'}
        
        try:
            # Test different K values
            k_values = [3, 5, 10, 15] if len(self.core.X_train_scaled_) > 20 else [3, 5]
            knn_results = {}
            
            # Split data for comparison
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                self.core.X_train_scaled_, self.core.y_train_, 
                test_size=0.3, random_state=42
            )
            
            # Test our RadiusNeighbors model
            radius_performance = self._test_radius_model(X_train_split, X_val_split, y_train_split, y_val_split)
            
            # Test KNN models
            for k in k_values:
                knn_performance = self._test_knn_model(k, X_train_split, X_val_split, y_train_split, y_val_split)
                knn_results[f'knn_k_{k}'] = knn_performance
            
            # Find best KNN
            best_knn_analysis = self._find_best_knn(knn_results)
            
            # Neighborhood comparison
            neighborhood_comparison = self._compare_neighborhood_characteristics(X_val_split)
            
            return {
                'radius_neighbors_performance': radius_performance,
                'knn_results': knn_results,
                'best_knn_analysis': best_knn_analysis,
                'neighborhood_comparison': neighborhood_comparison,
                'comparison_summary': self._summarize_knn_comparison(radius_performance, best_knn_analysis)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def compare_with_global_methods(self) -> Dict[str, Any]:
        """
        Compare with global regression methods.
        
        Returns:
        --------
        Dict[str, Any]
            Comparison results with global methods
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before comparison'}
        
        try:
            # Split data for comparison
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                self.core.X_train_scaled_, self.core.y_train_, 
                test_size=0.3, random_state=42
            )
            
            # Test our RadiusNeighbors model
            radius_performance = self._test_radius_model(X_train_split, X_val_split, y_train_split, y_val_split)
            
            # Global methods to test
            global_methods = {
                'linear_regression': LinearRegression(),
                'decision_tree': DecisionTreeRegressor(random_state=42, max_depth=10),
                'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            }
            
            global_results = {}
            
            for method_name, model in global_methods.items():
                global_performance = self._test_global_method(
                    method_name, model, X_train_split, X_val_split, y_train_split, y_val_split
                )
                global_results[method_name] = global_performance
            
            # Performance ranking
            ranking_analysis = self._rank_algorithm_performance(radius_performance, global_results)
            
            return {
                'radius_neighbors_performance': radius_performance,
                'global_methods_results': global_results,
                'performance_ranking': ranking_analysis,
                'method_characteristics': self._analyze_method_characteristics(global_results),
                'recommendation': self._recommend_best_method(radius_performance, global_results)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_metric_comparison(self) -> Dict[str, Any]:
        """
        Compare different distance metrics for the radius neighbors algorithm.
        
        Returns:
        --------
        Dict[str, Any]
            Metric comparison analysis
        """
        if not self.core.is_fitted_:
            return {'error': 'Model must be fitted before analysis'}
        
        try:
            metrics_to_test = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            metric_results = {}
            
            # Split data for testing
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                self.core.X_train_scaled_, self.core.y_train_, 
                test_size=0.3, random_state=42
            )
            
            for metric in metrics_to_test:
                metric_performance = self._test_distance_metric(
                    metric, X_train_split, X_val_split, y_train_split, y_val_split
                )
                metric_results[metric] = metric_performance
            
            # Find best metric
            best_metric_analysis = self._find_best_metric(metric_results)
            
            return {
                'metric_results': metric_results,
                'best_metric_analysis': best_metric_analysis,
                'current_metric': self.core.metric,
                'metric_characteristics': self._describe_metric_characteristics(),
                'recommendation': best_metric_analysis.get('best_metric', self.core.metric)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== PRIVATE ANALYSIS METHODS ====================
    
    def _analyze_radius_coverage(self) -> Dict[str, Any]:
        """Analyze how well the current radius covers the data space."""
        try:
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            
            coverage_percentage = (np.sum(neighbor_counts > 0) / len(neighbor_counts)) * 100
            isolated_points = np.sum(neighbor_counts == 1)  # Only themselves
            well_connected = np.sum(neighbor_counts >= 5)
            
            return {
                'coverage_percentage': float(coverage_percentage),
                'isolated_points': int(isolated_points),
                'well_connected_points': int(well_connected),
                'average_neighbors': float(np.mean(neighbor_counts)),
                'neighbor_count_distribution': {
                    'min': int(np.min(neighbor_counts)),
                    'max': int(np.max(neighbor_counts)),
                    'median': float(np.median(neighbor_counts)),
                    'std': float(np.std(neighbor_counts))
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_density_distribution(self) -> Dict[str, Any]:
        """Analyze the density distribution of the training data."""
        try:
            # Calculate local densities
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            densities = np.array([len(neighbors) for neighbors in neighbor_indices])
            
            # Statistical analysis
            density_stats = {
                'mean_density': float(np.mean(densities)),
                'density_variance': float(np.var(densities)),
                'density_skewness': float(self._calculate_skewness(densities)),
                'density_kurtosis': float(self._calculate_kurtosis(densities))
            }
            
            # Density regions
            low_density_threshold = np.percentile(densities, 25)
            high_density_threshold = np.percentile(densities, 75)
            
            density_regions = {
                'low_density_points': int(np.sum(densities <= low_density_threshold)),
                'medium_density_points': int(np.sum((densities > low_density_threshold) & (densities <= high_density_threshold))),
                'high_density_points': int(np.sum(densities > high_density_threshold)),
                'density_ratio': float(high_density_threshold / low_density_threshold) if low_density_threshold > 0 else 0
            }
            
            return {
                'density_statistics': density_stats,
                'density_regions': density_regions,
                'density_distribution': densities.tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_neighborhood_characteristics(self) -> Dict[str, Any]:
        """Analyze detailed neighborhood characteristics."""
        try:
            # Get distances and indices
            distances, indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_)
            
            neighbor_count_stats = {}
            distance_stats = {}
            
            # Analyze by neighbor count groups
            neighbor_counts = np.array([len(neighbors) for neighbors in indices])
            
            for count_range in [(1, 3), (4, 10), (11, 20), (21, float('inf'))]:
                range_name = f"{count_range[0]}-{count_range[1] if count_range[1] != float('inf') else 'inf'}_neighbors"
                mask = (neighbor_counts >= count_range[0]) & (neighbor_counts <= count_range[1])
                
                if np.any(mask):
                    range_distances = [distances[i] for i in range(len(distances)) if mask[i]]
                    all_range_distances = np.concatenate([d for d in range_distances if len(d) > 0])
                    
                    if len(all_range_distances) > 0:
                        neighbor_count_stats[range_name] = {
                            'count': int(np.sum(mask)),
                            'percentage': float(np.sum(mask) / len(neighbor_counts) * 100)
                        }
                        
                        distance_stats[range_name] = {
                            'mean_distance': float(np.mean(all_range_distances)),
                            'max_distance': float(np.max(all_range_distances)),
                            'distance_std': float(np.std(all_range_distances))
                        }
            
            return {
                'neighbor_count_stats': neighbor_count_stats,
                'distance_stats': distance_stats,
                'overall_stats': {
                    'total_points': len(neighbor_counts),
                    'mean_neighbors': float(np.mean(neighbor_counts)),
                    'median_neighbors': float(np.median(neighbor_counts))
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_distance_metrics(self) -> Dict[str, Any]:
        """Analyze the impact of different distance metrics."""
        try:
            metrics_comparison = {}
            current_metric = self.core.metric
            
            # Test subset of data for efficiency
            test_size = min(50, len(self.core.X_train_scaled_))
            test_indices = np.random.choice(len(self.core.X_train_scaled_), test_size, replace=False)
            X_test_subset = self.core.X_train_scaled_[test_indices]
            
            for metric in ['euclidean', 'manhattan', 'chebyshev']:
                try:
                    temp_model = RadiusNeighborsRegressor(
                        radius=self.core.effective_radius_,
                        metric=metric,
                        weights=self.core.weights
                    )
                    temp_model.fit(self.core.X_train_scaled_, self.core.y_train_)
                    
                    # Analyze neighborhood characteristics with this metric
                    neighbor_indices = temp_model.radius_neighbors(X_test_subset, return_distance=False)
                    neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
                    
                    metrics_comparison[metric] = {
                        'mean_neighbors': float(np.mean(neighbor_counts)),
                        'coverage': float(np.sum(neighbor_counts > 0) / len(neighbor_counts) * 100),
                        'isolated_points': int(np.sum(neighbor_counts == 1))
                    }
                    
                except Exception as metric_error:
                    metrics_comparison[metric] = {'error': str(metric_error)}
            
            return {
                'current_metric': current_metric,
                'metrics_comparison': metrics_comparison,
                'recommendation': self._recommend_best_distance_metric(metrics_comparison)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_optimal_radius(self) -> Dict[str, Any]:
        """Analyze optimal radius values for the current dataset."""
        try:
            radius_values = np.logspace(-1, 1, 10)  # Test from 0.1 to 10
            radius_analysis = {}
            
            # Use subset for efficiency
            X_subset, _, y_subset, _ = train_test_split(
                self.core.X_train_scaled_, self.core.y_train_, 
                test_size=0.7, random_state=42
            )
            
            for radius in radius_values:
                try:
                    temp_model = RadiusNeighborsRegressor(
                        radius=radius,
                        weights=self.core.weights,
                        metric=self.core.metric
                    )
                    temp_model.fit(X_subset, y_subset)
                    
                    # Quick performance evaluation
                    predictions = temp_model.predict(X_subset)
                    r2 = r2_score(y_subset, predictions)
                    
                    # Neighborhood analysis
                    neighbor_indices = temp_model.radius_neighbors(X_subset, return_distance=False)
                    neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
                    coverage = np.sum(neighbor_counts > 0) / len(neighbor_counts) * 100
                    
                    radius_analysis[f'radius_{radius:.2f}'] = {
                        'r2_score': float(r2),
                        'coverage': float(coverage),
                        'mean_neighbors': float(np.mean(neighbor_counts)),
                        'isolated_points': int(np.sum(neighbor_counts == 1))
                    }
                    
                except Exception as radius_error:
                    radius_analysis[f'radius_{radius:.2f}'] = {'error': str(radius_error)}
            
            # Find optimal radius
            valid_results = {k: v for k, v in radius_analysis.items() if 'error' not in v}
            if valid_results:
                best_radius = max(valid_results.keys(), key=lambda k: valid_results[k]['r2_score'])
                optimal_radius_value = float(best_radius.split('_')[1])
            else:
                optimal_radius_value = self.core.effective_radius_
                best_radius = f'radius_{optimal_radius_value:.2f}'
            
            return {
                'current_radius': self.core.effective_radius_,
                'radius_analysis': radius_analysis,
                'optimal_radius': optimal_radius_value,
                'improvement_potential': valid_results.get(best_radius, {}).get('r2_score', 0) - 
                                       valid_results.get(f'radius_{self.core.effective_radius_:.2f}', {}).get('r2_score', 0)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_outlier_behavior(self) -> Dict[str, Any]:
        """Analyze how the algorithm handles outliers."""
        try:
            # Calculate isolation scores
            neighbor_indices = self.core.model_.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            isolation_scores = self._calculate_isolation_scores(neighbor_indices)
            
            # Identify potential outliers
            outlier_threshold = np.percentile(isolation_scores, 95)
            outlier_indices = np.where(isolation_scores >= outlier_threshold)[0]
            
            # Analyze outlier characteristics
            outlier_analysis = {
                'total_outliers': len(outlier_indices),
                'outlier_percentage': float(len(outlier_indices) / len(isolation_scores) * 100),
                'mean_isolation_score': float(np.mean(isolation_scores)),
                'outlier_threshold': float(outlier_threshold)
            }
            
            # Outlier impact on predictions
            if len(outlier_indices) > 0:
                outlier_targets = self.core.y_train_[outlier_indices]
                non_outlier_targets = np.delete(self.core.y_train_, outlier_indices)
                
                outlier_analysis.update({
                    'outlier_target_mean': float(np.mean(outlier_targets)),
                    'non_outlier_target_mean': float(np.mean(non_outlier_targets)),
                    'target_difference': float(np.mean(outlier_targets) - np.mean(non_outlier_targets))
                })
            
            return outlier_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_baseline_performance(self) -> Dict[str, Any]:
        """Get baseline performance metrics for feature importance analysis."""
        try:
            # Split data for validation
            train_idx, val_idx = train_test_split(
                range(len(self.core.X_train_scaled_)), test_size=0.3, random_state=42
            )
            
            X_train_base = self.core.X_train_scaled_[train_idx]
            X_val_base = self.core.X_train_scaled_[val_idx]
            y_train_base = self.core.y_train_[train_idx]
            y_val_base = self.core.y_train_[val_idx]
            
            # Create baseline model
            baseline_model = RadiusNeighborsRegressor(
                radius=self.core.effective_radius_,
                weights=self.core.weights,
                algorithm=self.core.algorithm,
                metric=self.core.metric,
                p=self.core.p
            )
            
            baseline_model.fit(X_train_base, y_train_base)
            y_pred_base = baseline_model.predict(X_val_base)
            
            # Calculate metrics
            baseline_r2 = r2_score(y_val_base, y_pred_base)
            baseline_mse = mean_squared_error(y_val_base, y_pred_base)
            baseline_mae = mean_absolute_error(y_val_base, y_pred_base)
            
            # Neighborhood metrics
            neighbor_indices = baseline_model.radius_neighbors(X_val_base, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            coverage = ((len(neighbor_counts) - np.sum(neighbor_counts == 0)) / len(neighbor_counts)) * 100
            
            return {
                'r2': baseline_r2,
                'mse': baseline_mse,
                'mae': baseline_mae,
                'coverage': coverage,
                'mean_neighbors': np.mean(neighbor_counts)
            }
            
        except Exception as e:
            return {'r2': 0, 'mse': float('inf'), 'mae': float('inf'), 'coverage': 0, 'mean_neighbors': 0}
    
    def _analyze_feature_permutation(self, feature_index: int) -> Dict[str, Any]:
        """Analyze performance after permuting a specific feature."""
        try:
            # Create permuted dataset
            X_permuted = self.core.X_train_scaled_.copy()
            np.random.shuffle(X_permuted[:, feature_index])
            
            # Split data
            train_idx, val_idx = train_test_split(
                range(len(X_permuted)), test_size=0.3, random_state=42
            )
            
            X_train_perm = X_permuted[train_idx]
            X_val_perm = X_permuted[val_idx]
            y_train_perm = self.core.y_train_[train_idx]
            y_val_perm = self.core.y_train_[val_idx]
            
            # Train model with permuted feature
            perm_model = RadiusNeighborsRegressor(
                radius=self.core.effective_radius_,
                weights=self.core.weights,
                algorithm=self.core.algorithm,
                metric=self.core.metric,
                p=self.core.p
            )
            
            perm_model.fit(X_train_perm, y_train_perm)
            y_pred_perm = perm_model.predict(X_val_perm)
            
            return {
                'r2': r2_score(y_val_perm, y_pred_perm),
                'mse': mean_squared_error(y_val_perm, y_pred_perm),
                'mae': mean_absolute_error(y_val_perm, y_pred_perm)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_feature_interactions(self) -> Dict[str, Any]:
        """Analyze how features interact in determining neighborhoods."""
        try:
            if self.core.n_features_in_ < 2:
                return {'error': 'Need at least 2 features for interaction analysis'}
            
            interactions = {}
            
            # Analyze pairwise feature correlations in neighborhood formation
            max_features = min(5, self.core.n_features_in_)  # Limit for efficiency
            for i in range(max_features):
                for j in range(i+1, max_features):
                    feature_i_name = self.core.feature_names_[i]
                    feature_j_name = self.core.feature_names_[j]
                    
                    # Calculate distance correlation in 2D subspace
                    X_pair = self.core.X_train_scaled_[:, [i, j]]
                    
                    # Create temporary model with just these 2 features
                    temp_model = RadiusNeighborsRegressor(
                        radius=self.core.effective_radius_,
                        metric=self.core.metric
                    )
                    temp_model.fit(X_pair, self.core.y_train_)
                    
                    # Analyze neighborhood formation
                    neighbor_indices = temp_model.radius_neighbors(X_pair, return_distance=False)
                    neighbor_counts_pair = np.array([len(neighbors) for neighbors in neighbor_indices])
                    
                    interactions[f'{feature_i_name}_vs_{feature_j_name}'] = {
                        'mean_neighbors_2d': float(np.mean(neighbor_counts_pair)),
                        'correlation': float(np.corrcoef(self.core.X_train_scaled_[:, i], self.core.X_train_scaled_[:, j])[0, 1]),
                        'coverage_2d': float(((len(neighbor_counts_pair) - np.sum(neighbor_counts_pair == 0)) / len(neighbor_counts_pair)) * 100)
                    }
            
            return interactions
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_dimensionality_impact(self) -> Dict[str, Any]:
        """Analyze how dimensionality affects radius-based neighborhoods."""
        try:
            dimensionality_results = {}
            
            # Test with different numbers of features (dimensionality reduction)
            max_dims = min(self.core.n_features_in_, 10)  # Test up to 10 dimensions
            dims_to_test = [1, 2, 3, 5] if max_dims >= 5 else list(range(1, max_dims + 1))
            
            for n_dims in dims_to_test:
                if n_dims <= self.core.n_features_in_:
                    # Use first n_dims features
                    X_reduced = self.core.X_train_scaled_[:, :n_dims]
                    
                    # Create model with reduced dimensionality
                    temp_model = RadiusNeighborsRegressor(
                        radius=self.core.effective_radius_,
                        metric=self.core.metric
                    )
                    temp_model.fit(X_reduced, self.core.y_train_)
                    
                    # Analyze neighborhood characteristics
                    neighbor_indices = temp_model.radius_neighbors(X_reduced, return_distance=False)
                    neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
                    
                    coverage = ((len(neighbor_counts) - np.sum(neighbor_counts == 0)) / len(neighbor_counts)) * 100
                    
                    dimensionality_results[f'{n_dims}d'] = {
                        'mean_neighbors': float(np.mean(neighbor_counts)),
                        'coverage': float(coverage),
                        'zero_neighbor_points': int(np.sum(neighbor_counts == 0)),
                        'curse_of_dimensionality_effect': float(np.mean(neighbor_counts) / (np.pi * self.core.effective_radius_**n_dims))
                    }
            
            return dimensionality_results
            
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== UTILITY METHODS ====================
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        try:
            return stats.skew(data)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution."""
        try:
            return stats.kurtosis(data)
        except:
            return 0.0
    
    def _calculate_isolation_scores(self, neighbor_indices: List[np.ndarray]) -> np.ndarray:
        """Calculate isolation scores for all points."""
        neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
        # Isolation score: inverse of neighbor count (higher = more isolated)
        isolation_scores = 1.0 / (neighbor_counts + 1e-8)  # Small epsilon to avoid division by zero
        return isolation_scores
    
    def _count_neighbors_at_radius(self, radius: float) -> np.ndarray:
        """Count neighbors at a specific radius."""
        try:
            temp_model = RadiusNeighborsRegressor(radius=radius, metric=self.core.metric)
            temp_model.fit(self.core.X_train_scaled_, self.core.y_train_)
            neighbor_indices = temp_model.radius_neighbors(self.core.X_train_scaled_, return_distance=False)
            return np.array([len(neighbors) for neighbors in neighbor_indices])
        except:
            return np.zeros(len(self.core.X_train_scaled_))
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for performance profiling."""
        try:
            return {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': platform.python_version()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _profile_data_characteristics(self) -> Dict[str, Any]:
        """Profile characteristics of the training data."""
        return {
            'n_samples': self.core.n_samples_in_,
            'n_features': self.core.n_features_in_,
            'data_density': self.core.n_samples_in_ / self.core.n_features_in_ if self.core.n_features_in_ > 0 else 0,
            'feature_variance': float(np.mean(np.var(self.core.X_train_scaled_, axis=0))),
            'target_variance': float(np.var(self.core.y_train_)),
            'effective_radius': self.core.effective_radius_,
            'scaling_applied': self.core.scaler_ is not None
        }
    
    def _profile_operations_timing(self) -> Dict[str, Any]:
        """Profile timing of various operations."""
        try:
            operations_timing = {}
            
            # Subset for timing tests
            test_size = min(100, len(self.core.X_train_scaled_))
            X_test = self.core.X_train_scaled_[:test_size]
            
            # Time neighbor search
            start_time = time.time()
            self.core.model_.radius_neighbors(X_test)
            operations_timing['neighbor_search'] = time.time() - start_time
            
            # Time prediction
            start_time = time.time()
            self.core.model_.predict(X_test)
            operations_timing['prediction'] = time.time() - start_time
            
            # Time fit (on subset)
            start_time = time.time()
            temp_model = RadiusNeighborsRegressor(radius=self.core.effective_radius_)
            temp_model.fit(X_test, self.core.y_train_[:test_size])
            operations_timing['fit'] = time.time() - start_time
            
            return operations_timing
            
        except Exception as e:
            return {'error': str(e)}
    
    def _profile_memory_usage(self) -> float:
        """Profile memory usage of the algorithm."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # MB
        except:
            return 0.0
    
    def _analyze_scalability(self) -> Dict[str, Any]:
        """Analyze algorithm scalability with different data sizes."""
        try:
            scalability_results = {}
            base_size = len(self.core.X_train_scaled_)
            
            # Test with different sample sizes
            test_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
            
            for size_fraction in test_sizes:
                n_samples = int(base_size * size_fraction)
                if n_samples >= 10:  # Minimum samples
                    
                    # Sample data
                    indices = np.random.choice(base_size, n_samples, replace=False)
                    X_sample = self.core.X_train_scaled_[indices]
                    y_sample = self.core.y_train_[indices]
                    
                    # Time fitting
                    start_time = time.time()
                    temp_model = RadiusNeighborsRegressor(radius=self.core.effective_radius_)
                    temp_model.fit(X_sample, y_sample)
                    fit_time = time.time() - start_time
                    
                    # Time prediction
                    start_time = time.time()
                    temp_model.predict(X_sample[:min(50, len(X_sample))])
                    pred_time = time.time() - start_time
                    
                    scalability_results[f'size_{int(size_fraction*100)}%'] = {
                        'n_samples': n_samples,
                        'fit_time': fit_time,
                        'prediction_time': pred_time,
                        'time_per_sample': fit_time / n_samples if n_samples > 0 else 0
                    }
            
            return scalability_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _profile_algorithm_timing(self) -> Dict[str, Any]:
        """Profile timing of different algorithm options."""
        try:
            algorithm_timing = {}
            algorithms = ['ball_tree', 'brute']  # Skip 'kd_tree' for high-dimensional data
            
            test_size = min(100, len(self.core.X_train_scaled_))
            X_test = self.core.X_train_scaled_[:test_size]
            y_test = self.core.y_train_[:test_size]
            
            for algorithm in algorithms:
                try:
                    start_time = time.time()
                    temp_model = RadiusNeighborsRegressor(
                        radius=self.core.effective_radius_,
                        algorithm=algorithm
                    )
                    temp_model.fit(X_test, y_test)
                    temp_model.predict(X_test[:10])
                    algorithm_timing[algorithm] = time.time() - start_time
                    
                except Exception:
                    algorithm_timing[algorithm] = float('inf')
            
            return algorithm_timing
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_efficiency_metrics(self, operations_timing: Dict[str, Any], memory_usage: float) -> Dict[str, Any]:
        """Calculate efficiency metrics."""
        try:
            fit_time = operations_timing.get('fit', 0)
            pred_time = operations_timing.get('prediction', 0)
            neighbor_time = operations_timing.get('neighbor_search', 0)
            
            return {
                'fit_time_per_sample': fit_time / self.core.n_samples_in_ if self.core.n_samples_in_ > 0 else 0,
                'prediction_time_per_sample': pred_time / 100 if pred_time > 0 else 0,  # Assuming 100 predictions
                'memory_efficiency': memory_usage / self.core.n_samples_in_ if self.core.n_samples_in_ > 0 else 0,
                'neighbor_search_efficiency': neighbor_time / 100 if neighbor_time > 0 else 0,
                'radius_efficiency': self.core.effective_radius_ / np.mean(self._count_neighbors_at_radius(self.core.effective_radius_))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_performance_recommendations(self, timing: Dict[str, Any], efficiency: Dict[str, Any], algorithm_timing: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        try:
            # Algorithm recommendations
            if 'brute' in algorithm_timing and 'ball_tree' in algorithm_timing:
                if algorithm_timing['brute'] > algorithm_timing['ball_tree'] * 2:
                    recommendations.append("Consider using 'ball_tree' algorithm for better performance")
            
            # Memory recommendations
            if efficiency.get('memory_efficiency', 0) > 1:  # More than 1MB per sample
                recommendations.append("High memory usage detected. Consider using smaller radius or data subsampling")
            
            # Scalability recommendations
            if efficiency.get('fit_time_per_sample', 0) > 0.001:  # More than 1ms per sample
                recommendations.append("Slow fitting detected. Consider using 'auto' algorithm or reducing data size")
            
            # Radius recommendations based on analysis
            if hasattr(self, 'analysis_cache_') and 'radius_behavior' in self.analysis_cache_:
                coverage = self.analysis_cache_['radius_behavior'].get('radius_coverage', {}).get('coverage_percentage', 0)
                if coverage < 70:
                    recommendations.append("Low coverage detected. Consider increasing radius value")
                elif coverage > 95:
                    recommendations.append("Very high coverage. Consider decreasing radius to improve efficiency")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]
    
    def _analyze_bottlenecks(self, operations_timing: Dict[str, Any], efficiency: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        try:
            # Find slowest operation
            slowest_operation = max(operations_timing.keys(), 
                                  key=lambda k: operations_timing[k] 
                                  if isinstance(operations_timing[k], (int, float)) else 0)
            
            return {
                'slowest_operation': slowest_operation,
                'memory_intensive': efficiency.get('memory_efficiency', 0) > 1.0,  # > 1MB per sample
                'scalability_concern': efficiency.get('fit_time_per_sample', 0) > 0.01,  # > 10ms per sample
                'bottleneck_score': max(operations_timing.values()) if operations_timing else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== COMPARISON HELPER METHODS ====================
    
    def _test_radius_model(self, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Test RadiusNeighbors model performance."""
        try:
            radius_model = RadiusNeighborsRegressor(
                radius=self.core.effective_radius_,
                weights=self.core.weights,
                metric=self.core.metric,
                algorithm=self.core.algorithm
            )
            radius_model.fit(X_train, y_train)
            radius_pred = radius_model.predict(X_val)
            
            return {
                'r2_score': float(r2_score(y_val, radius_pred)),
                'mse': float(mean_squared_error(y_val, radius_pred)),
                'mae': float(mean_absolute_error(y_val, radius_pred)),
                'effective_radius': self.core.effective_radius_
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_knn_model(self, k: int, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Test KNN model performance."""
        try:
            knn_model = KNeighborsRegressor(
                n_neighbors=k,
                weights=self.core.weights,
                metric=self.core.metric,
                algorithm=self.core.algorithm
            )
            knn_model.fit(X_train, y_train)
            knn_pred = knn_model.predict(X_val)
            
            return {
                'r2_score': float(r2_score(y_val, knn_pred)),
                'mse': float(mean_squared_error(y_val, knn_pred)),
                'mae': float(mean_absolute_error(y_val, knn_pred)),
                'k_value': k
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_global_method(self, method_name: str, model: Any, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Test global method performance."""
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            return {
                'r2_score': float(r2_score(y_val, pred)),
                'mse': float(mean_squared_error(y_val, pred)),
                'mae': float(mean_absolute_error(y_val, pred)),
                'method_name': method_name
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_distance_metric(self, metric: str, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Test different distance metric performance."""
        try:
            model = RadiusNeighborsRegressor(
                radius=self.core.effective_radius_,
                metric=metric,
                weights=self.core.weights
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            return {
                'r2_score': float(r2_score(y_val, pred)),
                'mse': float(mean_squared_error(y_val, pred)),
                'mae': float(mean_absolute_error(y_val, pred)),
                'metric': metric
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== ANALYSIS HELPER METHODS ====================
    
    def _analyze_cv_fold(self, fold: int, train_idx: np.ndarray, val_idx: np.ndarray) -> Dict[str, Any]:
        """Analyze a single cross-validation fold."""
        try:
            X_train_fold = self.core.X_train_scaled_[train_idx]
            X_val_fold = self.core.X_train_scaled_[val_idx]
            y_train_fold = self.core.y_train_[train_idx]
            y_val_fold = self.core.y_train_[val_idx]
            
            # Train model on fold
            fold_model = RadiusNeighborsRegressor(
                radius=self.core.effective_radius_,
                weights=self.core.weights,
                metric=self.core.metric
            )
            fold_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = fold_model.predict(X_val_fold)
            
            # Calculate metrics
            r2 = r2_score(y_val_fold, y_pred_fold)
            mse = mean_squared_error(y_val_fold, y_pred_fold)
            
            # Neighborhood analysis
            neighbor_indices = fold_model.radius_neighbors(X_val_fold, return_distance=False)
            neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_indices])
            coverage = np.sum(neighbor_counts > 0) / len(neighbor_counts) * 100
            
            return {
                'fold': fold,
                'r2_score': float(r2),
                'mse': float(mse),
                'coverage': float(coverage),
                'mean_neighbors': float(np.mean(neighbor_counts)),
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            }
            
        except Exception as e:
            return {'fold': fold, 'error': str(e)}
    
    def _analyze_adaptive_behavior(self) -> Dict[str, Any]:
        """Analyze adaptive radius behavior across CV folds."""
        try:
            if not self.core.adaptive_radius:
                return {'adaptive_radius_enabled': False}
            
            # Analyze how adaptive radius would behave on different data subsets
            adaptive_results = {}
            
            # Test on different data densities
            for density_level in ['low', 'medium', 'high']:
                # Create density-based subset
                if density_level == 'low':
                    subset_indices = np.random.choice(len(self.core.X_train_scaled_), 
                                                    size=len(self.core.X_train_scaled_)//3, replace=False)
                elif density_level == 'medium':
                    subset_indices = np.random.choice(len(self.core.X_train_scaled_), 
                                                    size=len(self.core.X_train_scaled_)//2, replace=False)
                else:  # high
                    subset_indices = np.arange(len(self.core.X_train_scaled_))
                
                X_subset = self.core.X_train_scaled_[subset_indices]
                y_subset = self.core.y_train_[subset_indices]
                
                # Calculate what adaptive radius would be
                n_samples = len(X_subset)
                n_features = X_subset.shape[1]
                data_range = np.ptp(X_subset, axis=0).mean()
                volume_estimate = data_range ** n_features
                density_estimate = n_samples / volume_estimate if volume_estimate > 0 else 1.0
                density_factor = min(2.0, max(0.5, 1.0 / np.sqrt(density_estimate)))
                adaptive_radius = self.core.radius * density_factor
                
                adaptive_results[density_level] = {
                    'n_samples': n_samples,
                    'data_density': float(density_estimate),
                    'adaptive_radius': float(adaptive_radius),
                    'radius_factor': float(density_factor)
                }
            
            return {
                'adaptive_radius_enabled': True,
                'density_analysis': adaptive_results,
                'current_effective_radius': self.core.effective_radius_
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _find_best_knn(self, knn_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing KNN configuration."""
        try:
            valid_knn = {k: v for k, v in knn_results.items() if 'error' not in v}
            
            if not valid_knn:
                return {'best_knn': 'none', 'best_r2': 0}
            
            best_knn = max(valid_knn.keys(), key=lambda k: valid_knn[k]['r2_score'])
            best_r2 = valid_knn[best_knn]['r2_score']
            
            return {
                'best_knn_method': best_knn,
                'best_knn_r2': float(best_r2),
                'best_knn_config': valid_knn[best_knn],
                'performance_ranking': sorted(valid_knn.items(), key=lambda x: x[1]['r2_score'], reverse=True)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _compare_neighborhood_characteristics(self, X_val: np.ndarray) -> Dict[str, Any]:
        """Compare neighborhood characteristics between RadiusNeighbors and KNN."""
        try:
            # RadiusNeighbors neighborhoods
            radius_neighbors = self.core.model_.radius_neighbors(X_val, return_distance=False)
            radius_counts = np.array([len(neighbors) for neighbors in radius_neighbors])
            
            # KNN neighborhoods (k=5 for comparison)
            from sklearn.neighbors import NearestNeighbors
            knn_model = NearestNeighbors(n_neighbors=5, metric=self.core.metric)
            knn_model.fit(self.core.X_train_scaled_)
            knn_distances, knn_indices = knn_model.kneighbors(X_val)
            
            return {
                'radius_neighbors': {
                    'mean_neighbors': float(np.mean(radius_counts)),
                    'variable_neighbors': True,
                    'coverage': float(np.sum(radius_counts > 0) / len(radius_counts) * 100)
                },
                'knn_neighbors': {
                    'mean_neighbors': 5.0,  # Fixed
                    'variable_neighbors': False,
                    'mean_distance_to_5th': float(np.mean(knn_distances[:, -1]))
                },
                'comparison': {
                    'radius_more_adaptive': True,
                    'knn_more_consistent': True
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _summarize_knn_comparison(self, radius_performance: Dict[str, Any], best_knn_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the comparison between RadiusNeighbors and KNN."""
        try:
            radius_r2 = radius_performance.get('r2_score', 0)
            best_knn_r2 = best_knn_analysis.get('best_knn_r2', 0)
            
            return {
                'radius_vs_best_knn': float(radius_r2 - best_knn_r2),
                'radius_is_better': radius_r2 > best_knn_r2,
                'performance_difference_percentage': float((radius_r2 - best_knn_r2) / best_knn_r2 * 100) if best_knn_r2 > 0 else 0,
                'recommendation': 'radius_neighbors' if radius_r2 > best_knn_r2 else 'knn'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _rank_algorithm_performance(self, radius_performance: Dict[str, Any], global_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank algorithm performance against global methods."""
        try:
            performance_scores = {'radius_neighbors': radius_performance.get('r2_score', 0)}
            
            for method, result in global_results.items():
                if 'error' not in result:
                    performance_scores[method] = result.get('r2_score', 0)
            
            # Sort by performance
            ranking = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
            radius_rank = next((i+1 for i, (method, _) in enumerate(ranking) if method == 'radius_neighbors'), len(ranking))
            
            return {
                'performance_ranking': ranking,
                'radius_neighbors_rank': radius_rank,
                'total_methods': len(ranking),
                'performance_percentile': float((len(ranking) - radius_rank + 1) / len(ranking) * 100)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_method_characteristics(self, global_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of different methods."""
        method_characteristics = {
            'linear_regression': {
                'type': 'Global Linear',
                'complexity': 'Low',
                'interpretability': 'High',
                'assumptions': 'Linear relationship'
            },
            'decision_tree': {
                'type': 'Global Non-linear',
                'complexity': 'Medium',
                'interpretability': 'Medium',
                'assumptions': 'Hierarchical splits'
            },
            'random_forest': {
                'type': 'Global Ensemble',
                'complexity': 'High',
                'interpretability': 'Low',
                'assumptions': 'Ensemble of trees'
            },
            'radius_neighbors': {
                'type': 'Local Non-parametric',
                'complexity': 'Medium',
                'interpretability': 'Medium',
                'assumptions': 'Local similarity'
            }
        }
        
        # Add performance information
        for method, characteristics in method_characteristics.items():
            if method == 'radius_neighbors':
                continue
            elif method in global_results and 'error' not in global_results[method]:
                characteristics['r2_score'] = global_results[method].get('r2_score', 0)
            else:
                characteristics['r2_score'] = 0
        
        return method_characteristics
    
    def _recommend_best_method(self, radius_performance: Dict[str, Any], global_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend the best method based on performance and characteristics."""
        try:
            radius_r2 = radius_performance.get('r2_score', 0)
            
            # Find best global method
            best_global_method = None
            best_global_r2 = 0
            
            for method, result in global_results.items():
                if 'error' not in result:
                    r2 = result.get('r2_score', 0)
                    if r2 > best_global_r2:
                        best_global_r2 = r2
                        best_global_method = method
            
            # Performance comparison threshold
            performance_threshold = 0.05  # 5% difference threshold
            
            if radius_r2 > best_global_r2 + performance_threshold:
                recommendation = {
                    'recommended_method': 'radius_neighbors',
                    'reason': 'RadiusNeighbors significantly outperforms global methods',
                    'performance_advantage': float(radius_r2 - best_global_r2),
                    'confidence': 'high'
                }
            elif radius_r2 > best_global_r2 - performance_threshold:
                recommendation = {
                    'recommended_method': 'radius_neighbors',
                    'reason': 'RadiusNeighbors performs competitively with local adaptability benefits',
                    'performance_advantage': float(radius_r2 - best_global_r2),
                    'confidence': 'medium'
                }
            else:
                recommendation = {
                    'recommended_method': best_global_method,
                    'reason': f'{best_global_method} shows better performance',
                    'performance_advantage': float(best_global_r2 - radius_r2),
                    'confidence': 'high'
                }
            
            return recommendation
            
        except Exception as e:
            return {
                'recommended_method': 'radius_neighbors',
                'reason': f'Error in analysis: {str(e)}',
                'confidence': 'low'
            }
    
    def _find_best_metric(self, metric_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing distance metric."""
        try:
            valid_metrics = {k: v for k, v in metric_results.items() if 'error' not in v}
            
            if not valid_metrics:
                return {'best_metric': self.core.metric, 'best_r2': 0}
            
            best_metric = max(valid_metrics.keys(), key=lambda k: valid_metrics[k]['r2_score'])
            best_r2 = valid_metrics[best_metric]['r2_score']
            
            return {
                'best_metric': best_metric,
                'best_r2': float(best_r2),
                'best_config': valid_metrics[best_metric],
                'performance_ranking': sorted(valid_metrics.items(), key=lambda x: x[1]['r2_score'], reverse=True),
                'improvement_over_current': float(best_r2 - valid_metrics.get(self.core.metric, {}).get('r2_score', 0))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _describe_metric_characteristics(self) -> Dict[str, Any]:
        """Describe characteristics of different distance metrics."""
        return {
            'euclidean': {
                'description': 'Standard straight-line distance',
                'sensitivity': 'Equal sensitivity to all dimensions',
                'best_for': 'Continuous features with similar scales'
            },
            'manhattan': {
                'description': 'Sum of absolute differences',
                'sensitivity': 'Less sensitive to outliers',
                'best_for': 'High-dimensional sparse data'
            },
            'chebyshev': {
                'description': 'Maximum difference across dimensions',
                'sensitivity': 'Dominated by single largest difference',
                'best_for': 'When any single feature difference is critical'
            },
            'minkowski': {
                'description': 'Generalized distance metric (p-norm)',
                'sensitivity': 'Adjustable via p parameter',
                'best_for': 'When you want to tune distance sensitivity'
            }
        }
    
    def _recommend_best_distance_metric(self, metrics_comparison: Dict[str, Any]) -> str:
        """Recommend the best distance metric based on analysis."""
        try:
            # Find metric with best coverage and neighbor distribution
            best_metric = None
            best_score = 0
            
            for metric, results in metrics_comparison.items():
                if 'error' not in results:
                    # Score based on coverage and mean neighbors
                    coverage = results.get('coverage', 0)
                    mean_neighbors = results.get('mean_neighbors', 0)
                    isolated = results.get('isolated_points', float('inf'))
                    
                    # Calculate composite score (higher is better)
                    score = coverage * 0.6 + min(mean_neighbors * 10, 50) * 0.3 - isolated * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_metric = metric
            
            return best_metric if best_metric else self.core.metric
            
        except Exception:
            return self.core.metric
    
    def _get_top_features(self, importance_scores: Dict[str, float], n_features: int = 3) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        try:
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_features[:n_features]
        except Exception:
            return []


# ==================== TESTING FUNCTIONS ====================

def test_analysis_functionality():
    """Test the analysis functionality of RadiusNeighborsAnalysis."""
    print("🧪 Testing Radius Neighbors Analysis Functionality...")
    
    try:
        # Import core for testing
        from .radius_neighbors_core import RadiusNeighborsCore
        
        # Generate test data
        np.random.seed(42)
        n_samples, n_features = 150, 4
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(n_samples)
        
        # Test 1: Create and fit core model
        print("✅ Test 1: Core model setup")
        core = RadiusNeighborsCore(radius=1.5, auto_scale=True)
        training_results = core.train_model(X, y)
        assert training_results['model_fitted'] == True
        
        # Test 2: Initialize analysis component
        print("✅ Test 2: Analysis component initialization")
        analysis = RadiusNeighborsAnalysis(core)
        assert analysis.core == core
        
        # Test 3: Radius behavior analysis
        print("✅ Test 3: Radius behavior analysis")
        radius_analysis = analysis.analyze_radius_behavior()
        assert 'radius_coverage' in radius_analysis
        assert 'density_distribution' in radius_analysis
        
        # Test 4: Cross-validation analysis
        print("✅ Test 4: Cross-validation analysis")
        cv_analysis = analysis.analyze_cross_validation(cv_folds=3)
        assert 'cv_scores' in cv_analysis
        assert 'mean_cv_score' in cv_analysis
        
        # Test 5: Feature importance analysis
        print("✅ Test 5: Feature importance analysis")
        feature_analysis = analysis.analyze_feature_importance()
        assert 'feature_importance_scores' in feature_analysis
        assert 'baseline_performance' in feature_analysis
        
        # Test 6: Performance profiling
        print("✅ Test 6: Performance profiling")
        performance_profile = analysis.profile_performance()
        assert 'system_info' in performance_profile
        assert 'operations_timing' in performance_profile
        
        # Test 7: KNN comparison
        print("✅ Test 7: KNN comparison")
        knn_comparison = analysis.compare_with_knn()
        assert 'radius_neighbors_performance' in knn_comparison
        assert 'knn_results' in knn_comparison
        
        # Test 8: Global methods comparison
        print("✅ Test 8: Global methods comparison")
        global_comparison = analysis.compare_with_global_methods()
        assert 'radius_neighbors_performance' in global_comparison
        assert 'global_methods_results' in global_comparison
        
        # Test 9: Metric comparison
        print("✅ Test 9: Distance metric comparison")
        metric_comparison = analysis.analyze_metric_comparison()
        assert 'metric_results' in metric_comparison
        assert 'current_metric' in metric_comparison
        
        # Test 10: Cache functionality
        print("✅ Test 10: Analysis caching")
        assert 'radius_behavior' in analysis.analysis_cache_
        assert 'performance_profile' in analysis.analysis_cache_
        
        print("🎉 All analysis functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Analysis test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis_edge_cases():
    """Test edge cases for analysis component."""
    print("🔍 Testing Analysis Edge Cases...")
    
    try:
        from .radius_neighbors_core import RadiusNeighborsCore
        
        # Test 1: Unfitted model
        print("✅ Test 1: Unfitted model handling")
        core = RadiusNeighborsCore()
        analysis = RadiusNeighborsAnalysis(core)
        
        result = analysis.analyze_radius_behavior()
        assert 'error' in result
        
        # Test 2: Small dataset
        print("✅ Test 2: Small dataset handling")
        X_small = np.random.randn(10, 2)
        y_small = np.random.randn(10)
        
        core_small = RadiusNeighborsCore(radius=0.5)
        core_small.train_model(X_small, y_small)
        analysis_small = RadiusNeighborsAnalysis(core_small)
        
        cv_result = analysis_small.analyze_cross_validation(cv_folds=3)
        # Should handle small dataset gracefully
        
        # Test 3: Single feature dataset
        print("✅ Test 3: Single feature handling")
        X_single = np.random.randn(50, 1)
        y_single = X_single.flatten() + 0.1 * np.random.randn(50)
        
        core_single = RadiusNeighborsCore(radius=1.0)
        core_single.train_model(X_single, y_single)
        analysis_single = RadiusNeighborsAnalysis(core_single)
        
        feature_result = analysis_single.analyze_feature_importance()
        interaction_result = analysis_single._analyze_feature_interactions()
        assert 'error' in interaction_result  # Should detect insufficient features
        
        print("🎉 All edge case tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run tests when module is executed directly
    print("🚀 Running Radius Neighbors Analysis Tests...")
    
    # Test main functionality
    main_test = test_analysis_functionality()
    
    # Test edge cases
    edge_test = test_analysis_edge_cases()
    
    if main_test and edge_test:
        print("\n🎉 All tests passed! Analysis component is ready.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")