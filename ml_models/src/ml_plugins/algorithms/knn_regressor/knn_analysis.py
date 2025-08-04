"""
K-Nearest Neighbors Regressor Analysis Component
===============================================

This module provides comprehensive analysis capabilities for the K-Nearest Neighbors
Regressor algorithm, including parameter optimization, performance profiling,
and comparative analysis.

Features:
- K-value optimization and sensitivity analysis
- Distance metric and weighting strategy analysis
- Algorithm efficiency and scalability analysis
- Cross-validation and stability analysis
- Comparison with other regression methods
- Feature importance analysis for KNN
- Local density and neighborhood analysis

Author: Bachelor Thesis Project
Date: June 2025
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNNAnalysis:
    """
    Comprehensive analysis component for K-Nearest Neighbors Regressor.
    
    This class provides detailed analysis capabilities including parameter optimization,
    performance profiling, and comparative analysis with other methods.
    """
    
    def __init__(self, knn_core):
        """
        Initialize the KNN Analysis component.
        
        Parameters:
        -----------
        knn_core : KNNCore
            The core KNN component to analyze
        """
        self.core = knn_core
        
        # Analysis configuration
        self.config = {
            'cv_folds': 5,
            'cv_scoring': 'r2',
            'n_repeats': 5,
            'feature_importance_method': 'permutation',
            'compare_with_knn': True,
            'compare_with_global': True,
            'k_range': (1, 20),
            'n_jobs': 1,
            'random_state': 42,
            'verbose': False
        }
        
        # Analysis results cache
        self._analysis_cache = {}
        
        logger.info("✅ KNN Analysis component initialized")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the analysis component.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        try:
            self.config.update(config)
            
            # Clear cache when configuration changes
            self._analysis_cache.clear()
            
            logger.info("✅ KNN Analysis configuration updated")
            
        except Exception as e:
            logger.error(f"❌ Configuration failed: {str(e)}")
            raise
    
    # ==================== K-VALUE ANALYSIS ====================
    
    def analyze_k_optimization(self) -> Dict[str, Any]:
        """
        Comprehensive K-value optimization analysis.
        
        Returns:
        --------
        Dict[str, Any]
            Complete K optimization results
        """
        try:
            if not self.core.is_fitted:
                return {'error': 'Core model must be fitted before analysis'}
            
            cache_key = 'k_optimization'
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
            
            logger.info("🔍 Starting comprehensive K optimization analysis...")
            
            X, y = self.core.X_train_scaled_, self.core.y_train_
            k_range = self.config.get('k_range', (1, 20))
            k_min, k_max = k_range
            k_max = min(k_max, len(X) - 1)
            
            if k_max <= k_min:
                k_max = k_min + 10
            
            k_values = list(range(k_min, k_max + 1))
            
            # Core K optimization from core component
            basic_analysis = self.core.analyze_optimal_k(self.core.X_train_, self.core.y_train_, k_range)
            
            # Extended analysis
            results = {
                'basic_optimization': basic_analysis,
                'detailed_analysis': self._detailed_k_analysis(k_values, X, y),
                'bias_variance_analysis': self._k_bias_variance_analysis(k_values, X, y),
                'stability_analysis': self._k_stability_analysis(k_values, X, y),
                'computational_analysis': self._k_computational_analysis(k_values, X, y)
            }
            
            # Generate recommendations
            results['recommendations'] = self._generate_k_recommendations(results)
            
            # Cache results
            self._analysis_cache[cache_key] = results
            
            logger.info("✅ K optimization analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ K optimization analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _detailed_k_analysis(self, k_values: List[int], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detailed analysis for different K values."""
        try:
            results = {
                'k_values': k_values,
                'train_scores': [],
                'val_scores': [],
                'test_scores': [],
                'prediction_times': [],
                'neighbor_statistics': []
            }
            
            original_k = self.core.n_neighbors
            
            for k in k_values:
                # Create temporary model
                temp_model = KNeighborsRegressor(
                    n_neighbors=k,
                    weights=self.core.weights,
                    algorithm=self.core.algorithm,
                    metric=self.core.metric,
                    p=self.core.p if self.core.metric == 'minkowski' else 2
                )
                
                # Training score
                temp_model.fit(X, y)
                train_pred = temp_model.predict(X)
                train_score = r2_score(y, train_pred)
                results['train_scores'].append(train_score)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    temp_model, X, y,
                    cv=min(self.config.get('cv_folds', 5), len(X) // 3),
                    scoring=self.config.get('cv_scoring', 'r2')
                )
                results['val_scores'].append(cv_scores.mean())
                
                # Prediction time
                start_time = time.time()
                temp_model.predict(X[:min(50, len(X))])
                pred_time = time.time() - start_time
                results['prediction_times'].append(pred_time)
                
                # Neighbor statistics
                distances, indices = temp_model.kneighbors(X[:min(20, len(X))])
                neighbor_stats = {
                    'mean_distance': distances.mean(),
                    'std_distance': distances.std(),
                    'min_distance': distances.min(),
                    'max_distance': distances.max()
                }
                results['neighbor_statistics'].append(neighbor_stats)
            
            # Restore original K
            self.core.n_neighbors = original_k
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Detailed K analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _k_bias_variance_analysis(self, k_values: List[int], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze bias-variance tradeoff for different K values."""
        try:
            from sklearn.model_selection import train_test_split
            
            n_iterations = 20
            bias_scores = []
            variance_scores = []
            
            for k in k_values:
                predictions_per_sample = []
                
                for _ in range(n_iterations):
                    # Random train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=np.random.randint(0, 1000)
                    )
                    
                    if len(X_train) < k:
                        continue
                    
                    # Train model
                    temp_model = KNeighborsRegressor(
                        n_neighbors=k,
                        weights=self.core.weights,
                        algorithm=self.core.algorithm,
                        metric=self.core.metric
                    )
                    temp_model.fit(X_train, y_train)
                    
                    # Predict
                    pred = temp_model.predict(X_test)
                    predictions_per_sample.append(pred)
                
                if predictions_per_sample:
                    # Calculate bias and variance
                    predictions_array = np.array(predictions_per_sample)
                    mean_predictions = predictions_array.mean(axis=0)
                    
                    # Bias: average of (mean_prediction - true_value)²
                    bias = np.mean((mean_predictions - y_test) ** 2)
                    
                    # Variance: average variance of predictions
                    variance = np.mean(np.var(predictions_array, axis=0))
                    
                    bias_scores.append(bias)
                    variance_scores.append(variance)
                else:
                    bias_scores.append(float('nan'))
                    variance_scores.append(float('nan'))
            
            # Find optimal K for bias-variance tradeoff
            total_error = np.array(bias_scores) + np.array(variance_scores)
            valid_indices = ~np.isnan(total_error)
            
            if np.any(valid_indices):
                optimal_idx = np.argmin(total_error[valid_indices])
                optimal_k = np.array(k_values)[valid_indices][optimal_idx]
            else:
                optimal_k = k_values[len(k_values) // 2]
            
            return {
                'k_values': k_values,
                'bias_scores': bias_scores,
                'variance_scores': variance_scores,
                'total_error': total_error.tolist(),
                'optimal_k_bias_variance': optimal_k,
                'bias_variance_ratio': np.array(bias_scores) / (np.array(variance_scores) + 1e-10)
            }
            
        except Exception as e:
            logger.error(f"❌ Bias-variance analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _k_stability_analysis(self, k_values: List[int], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction stability for different K values."""
        try:
            from sklearn.model_selection import KFold
            
            stability_scores = []
            consistency_scores = []
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for k in k_values:
                fold_predictions = []
                fold_scores = []
                
                for train_idx, val_idx in kf.split(X):
                    if len(train_idx) < k:
                        continue
                        
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    temp_model = KNeighborsRegressor(
                        n_neighbors=k,
                        weights=self.core.weights,
                        algorithm=self.core.algorithm,
                        metric=self.core.metric
                    )
                    temp_model.fit(X_train, y_train)
                    
                    # Predict and score
                    pred = temp_model.predict(X_val)
                    score = r2_score(y_val, pred)
                    
                    fold_predictions.append(pred)
                    fold_scores.append(score)
                
                if fold_scores:
                    # Stability: coefficient of variation of scores
                    stability = np.std(fold_scores) / (np.mean(fold_scores) + 1e-10)
                    stability_scores.append(stability)
                    
                    # Consistency: average correlation between fold predictions
                    if len(fold_predictions) >= 2:
                        correlations = []
                        for i in range(len(fold_predictions)):
                            for j in range(i + 1, len(fold_predictions)):
                                if len(fold_predictions[i]) == len(fold_predictions[j]):
                                    corr = np.corrcoef(fold_predictions[i], fold_predictions[j])[0, 1]
                                    if not np.isnan(corr):
                                        correlations.append(corr)
                        consistency = np.mean(correlations) if correlations else 0
                    else:
                        consistency = 0
                    
                    consistency_scores.append(consistency)
                else:
                    stability_scores.append(float('inf'))
                    consistency_scores.append(0)
            
            # Find most stable K
            stability_array = np.array(stability_scores)
            valid_indices = ~np.isinf(stability_array)
            
            if np.any(valid_indices):
                most_stable_idx = np.argmin(stability_array[valid_indices])
                most_stable_k = np.array(k_values)[valid_indices][most_stable_idx]
            else:
                most_stable_k = k_values[len(k_values) // 2]
            
            return {
                'k_values': k_values,
                'stability_scores': stability_scores,
                'consistency_scores': consistency_scores,
                'most_stable_k': most_stable_k,
                'stability_ranking': np.argsort(stability_scores).tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Stability analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _k_computational_analysis(self, k_values: List[int], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze computational complexity for different K values."""
        try:
            training_times = []
            prediction_times = []
            memory_usage = []
            
            for k in k_values:
                # Create model
                temp_model = KNeighborsRegressor(
                    n_neighbors=k,
                    weights=self.core.weights,
                    algorithm=self.core.algorithm,
                    metric=self.core.metric
                )
                
                # Training time
                start_time = time.time()
                temp_model.fit(X, y)
                train_time = time.time() - start_time
                training_times.append(train_time)
                
                # Prediction time
                test_samples = X[:min(100, len(X))]
                start_time = time.time()
                temp_model.predict(test_samples)
                pred_time = time.time() - start_time
                prediction_times.append(pred_time)
                
                # Memory usage estimation (simplified)
                base_memory = X.nbytes + y.nbytes
                if self.core.algorithm == 'brute':
                    # Brute force doesn't build tree structure
                    estimated_memory = base_memory * 1.1
                else:
                    # Tree algorithms have overhead
                    estimated_memory = base_memory * (1 + np.log(len(X)) / 10)
                
                memory_usage.append(estimated_memory / (1024 * 1024))  # MB
            
            return {
                'k_values': k_values,
                'training_times': training_times,
                'prediction_times': prediction_times,
                'memory_usage_mb': memory_usage,
                'efficiency_scores': [1.0 / (t + 1e-10) for t in prediction_times],
                'scalability_factor': np.polyfit(k_values, prediction_times, 1)[0] if len(k_values) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Computational analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_k_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive K value recommendations."""
        try:
            recommendations = []
            
            # From basic optimization
            basic_opt = analysis_results.get('basic_optimization', {})
            if 'optimal_k' in basic_opt:
                recommendations.append({
                    'type': 'performance_optimal',
                    'k_value': basic_opt['optimal_k'],
                    'score': basic_opt.get('optimal_score', 0),
                    'confidence': basic_opt.get('recommendation', {}).get('confidence', 'medium'),
                    'reason': 'Highest cross-validation score'
                })
            
            # From bias-variance analysis
            bias_var = analysis_results.get('bias_variance_analysis', {})
            if 'optimal_k_bias_variance' in bias_var:
                recommendations.append({
                    'type': 'bias_variance_optimal',
                    'k_value': bias_var['optimal_k_bias_variance'],
                    'reason': 'Best bias-variance tradeoff'
                })
            
            # From stability analysis
            stability = analysis_results.get('stability_analysis', {})
            if 'most_stable_k' in stability:
                recommendations.append({
                    'type': 'stability_optimal',
                    'k_value': stability['most_stable_k'],
                    'reason': 'Most stable predictions'
                })
            
            # Rule of thumb
            n_samples = len(self.core.X_train_) if self.core.X_train_ is not None else 0
            if n_samples > 0:
                sqrt_rule_k = max(1, int(np.sqrt(n_samples)))
                recommendations.append({
                    'type': 'rule_of_thumb',
                    'k_value': sqrt_rule_k,
                    'reason': f'Square root rule for {n_samples} samples'
                })
            
            # Generate final recommendation
            if recommendations:
                # Prefer performance optimal if available and confident
                performance_recs = [r for r in recommendations if r['type'] == 'performance_optimal']
                if performance_recs and performance_recs[0].get('confidence') in ['high', 'medium']:
                    final_rec = performance_recs[0]
                else:
                    # Otherwise, use most common recommendation
                    k_values = [r['k_value'] for r in recommendations]
                    final_rec = {
                        'type': 'consensus',
                        'k_value': int(np.median(k_values)),
                        'reason': f'Consensus from {len(recommendations)} analysis methods'
                    }
            else:
                final_rec = {
                    'type': 'default',
                    'k_value': 5,
                    'reason': 'Default value (no analysis available)'
                }
            
            return {
                'all_recommendations': recommendations,
                'final_recommendation': final_rec,
                'analysis_methods_used': len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"❌ K recommendations generation failed: {str(e)}")
            return {'error': str(e)}
    
    # ==================== DISTANCE METRIC ANALYSIS ====================
    
    def analyze_distance_metrics(self) -> Dict[str, Any]:
        """
        Comprehensive distance metric analysis.
        
        Returns:
        --------
        Dict[str, Any]
            Distance metric analysis results
        """
        try:
            if not self.core.is_fitted:
                return {'error': 'Core model must be fitted before analysis'}
            
            cache_key = 'distance_metrics'
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
            
            logger.info("🔍 Starting distance metric analysis...")
            
            X, y = self.core.X_train_scaled_, self.core.y_train_
            
            # Core metric analysis from core component
            basic_analysis = self.core.analyze_distance_metrics(self.core.X_train_, self.core.y_train_)
            
            # Extended analysis
            results = {
                'basic_comparison': basic_analysis,
                'detailed_analysis': self._detailed_metric_analysis(X, y),
                'feature_scaling_impact': self._metric_scaling_analysis(X, y),
                'dimensionality_impact': self._metric_dimensionality_analysis(X, y),
                'robustness_analysis': self._metric_robustness_analysis(X, y)
            }
            
            # Generate recommendations
            results['recommendations'] = self._generate_metric_recommendations(results)
            
            # Cache results
            self._analysis_cache[cache_key] = results
            
            logger.info("✅ Distance metric analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Distance metric analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _detailed_metric_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Detailed analysis of distance metrics with statistical tests."""
        try:
            metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            results = {}
            
            for metric in metrics:
                try:
                    # Multiple CV runs for statistical significance
                    cv_scores_runs = []
                    for run in range(5):
                        temp_model = KNeighborsRegressor(
                            n_neighbors=self.core.n_neighbors,
                            weights=self.core.weights,
                            algorithm=self.core.algorithm,
                            metric=metric,
                            p=self.core.p if metric == 'minkowski' else 2
                        )
                        
                        cv_scores = cross_val_score(
                            temp_model, X, y,
                            cv=min(5, len(X) // 3),
                            scoring='r2'
                        )
                        cv_scores_runs.extend(cv_scores)
                    
                    # Statistical analysis
                    mean_score = np.mean(cv_scores_runs)
                    std_score = np.std(cv_scores_runs)
                    confidence_interval = stats.t.interval(
                        0.95, len(cv_scores_runs) - 1,
                        loc=mean_score,
                        scale=std_score / np.sqrt(len(cv_scores_runs))
                    )
                    
                    results[metric] = {
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'confidence_interval': confidence_interval,
                        'n_evaluations': len(cv_scores_runs),
                        'all_scores': cv_scores_runs
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed detailed analysis for metric {metric}: {str(e)}")
                    results[metric] = {'error': str(e)}
            
            # Statistical significance tests
            significance_tests = self._perform_metric_significance_tests(results)
            
            return {
                'metric_results': results,
                'significance_tests': significance_tests
            }
            
        except Exception as e:
            logger.error(f"❌ Detailed metric analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _metric_scaling_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze impact of feature scaling on different metrics."""
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            scalers = {
                'none': None,
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler()
            }
            
            metrics = ['euclidean', 'manhattan', 'chebyshev']
            results = {}
            
            for scaler_name, scaler in scalers.items():
                results[scaler_name] = {}
                
                # Prepare data
                if scaler is not None:
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X
                
                for metric in metrics:
                    try:
                        temp_model = KNeighborsRegressor(
                            n_neighbors=self.core.n_neighbors,
                            weights=self.core.weights,
                            algorithm=self.core.algorithm,
                            metric=metric
                        )
                        
                        cv_scores = cross_val_score(
                            temp_model, X_scaled, y,
                            cv=min(5, len(X) // 3),
                            scoring='r2'
                        )
                        
                        results[scaler_name][metric] = {
                            'mean_score': cv_scores.mean(),
                            'std_score': cv_scores.std()
                        }
                        
                    except Exception as e:
                        results[scaler_name][metric] = {'error': str(e)}
            
            # Find best scaler for each metric
            best_scalers = {}
            for metric in metrics:
                best_score = -np.inf
                best_scaler = 'none'
                
                for scaler_name in scalers.keys():
                    score = results[scaler_name].get(metric, {}).get('mean_score', -np.inf)
                    if score > best_score:
                        best_score = score
                        best_scaler = scaler_name
                
                best_scalers[metric] = {
                    'best_scaler': best_scaler,
                    'best_score': best_score
                }
            
            return {
                'scaling_results': results,
                'best_scalers': best_scalers,
                'scaling_impact': self._calculate_scaling_impact(results)
            }
            
        except Exception as e:
            logger.error(f"❌ Metric scaling analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _metric_dimensionality_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze metric performance vs dimensionality."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.feature_selection import SelectKBest, f_regression
            
            n_features = X.shape[1]
            metrics = ['euclidean', 'manhattan', 'chebyshev']
            
            # Test different dimensionalities
            if n_features > 5:
                dim_ratios = [0.25, 0.5, 0.75, 1.0]
                dimensions = [max(1, int(n_features * ratio)) for ratio in dim_ratios]
            else:
                dimensions = list(range(1, n_features + 1))
            
            results = {}
            
            for metric in metrics:
                results[metric] = {
                    'dimensions': dimensions,
                    'pca_scores': [],
                    'selection_scores': []
                }
                
                for dim in dimensions:
                    try:
                        # PCA dimensionality reduction
                        if dim < n_features:
                            pca = PCA(n_components=dim)
                            X_pca = pca.fit_transform(X)
                        else:
                            X_pca = X
                        
                        temp_model = KNeighborsRegressor(
                            n_neighbors=min(self.core.n_neighbors, len(X) - 1),
                            metric=metric
                        )
                        
                        cv_scores_pca = cross_val_score(
                            temp_model, X_pca, y,
                            cv=min(3, len(X) // 3),
                            scoring='r2'
                        )
                        results[metric]['pca_scores'].append(cv_scores_pca.mean())
                        
                        # Feature selection
                        if dim < n_features:
                            selector = SelectKBest(f_regression, k=dim)
                            X_selected = selector.fit_transform(X, y)
                        else:
                            X_selected = X
                        
                        cv_scores_sel = cross_val_score(
                            temp_model, X_selected, y,
                            cv=min(3, len(X) // 3),
                            scoring='r2'
                        )
                        results[metric]['selection_scores'].append(cv_scores_sel.mean())
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Dimensionality analysis failed for {metric}, dim={dim}: {str(e)}")
                        results[metric]['pca_scores'].append(np.nan)
                        results[metric]['selection_scores'].append(np.nan)
            
            # Calculate curse of dimensionality factors
            curse_factors = {}
            for metric in metrics:
                pca_scores = np.array(results[metric]['pca_scores'])
                valid_scores = pca_scores[~np.isnan(pca_scores)]
                
                if len(valid_scores) > 1:
                    # Calculate how much performance degrades with dimensionality
                    curse_factor = (valid_scores[0] - valid_scores[-1]) / len(valid_scores)
                    curse_factors[metric] = curse_factor
                else:
                    curse_factors[metric] = 0
            
            return {
                'dimensionality_results': results,
                'curse_of_dimensionality': curse_factors,
                'optimal_dimensions': self._find_optimal_dimensions(results)
            }
            
        except Exception as e:
            logger.error(f"❌ Dimensionality analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _metric_robustness_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze metric robustness to outliers and noise."""
        try:
            metrics = ['euclidean', 'manhattan', 'chebyshev']
            results = {}
            
            # Original performance
            baseline_scores = {}
            for metric in metrics:
                temp_model = KNeighborsRegressor(
                    n_neighbors=self.core.n_neighbors,
                    metric=metric
                )
                cv_scores = cross_val_score(temp_model, X, y, cv=3, scoring='r2')
                baseline_scores[metric] = cv_scores.mean()
            
            # Add noise and test robustness
            noise_levels = [0.1, 0.2, 0.3]
            
            for noise_level in noise_levels:
                # Add noise to features
                noise = np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
                X_noisy = X + noise
                
                for metric in metrics:
                    if metric not in results:
                        results[metric] = {
                            'noise_levels': noise_levels,
                            'noisy_scores': [],
                            'robustness_scores': []
                        }
                    
                    try:
                        temp_model = KNeighborsRegressor(
                            n_neighbors=self.core.n_neighbors,
                            metric=metric
                        )
                        cv_scores = cross_val_score(temp_model, X_noisy, y, cv=3, scoring='r2')
                        noisy_score = cv_scores.mean()
                        
                        results[metric]['noisy_scores'].append(noisy_score)
                        
                        # Robustness: how much performance is retained
                        robustness = noisy_score / baseline_scores[metric] if baseline_scores[metric] > 0 else 0
                        results[metric]['robustness_scores'].append(robustness)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Robustness test failed for {metric}, noise={noise_level}: {str(e)}")
                        results[metric]['noisy_scores'].append(np.nan)
                        results[metric]['robustness_scores'].append(np.nan)
            
            # Calculate overall robustness
            robustness_summary = {}
            for metric in metrics:
                if metric in results:
                    robust_scores = np.array(results[metric]['robustness_scores'])
                    valid_scores = robust_scores[~np.isnan(robust_scores)]
                    
                    if len(valid_scores) > 0:
                        robustness_summary[metric] = {
                            'mean_robustness': valid_scores.mean(),
                            'min_robustness': valid_scores.min(),
                            'robustness_std': valid_scores.std()
                        }
                    else:
                        robustness_summary[metric] = {'error': 'No valid robustness scores'}
            
            return {
                'baseline_scores': baseline_scores,
                'noise_analysis': results,
                'robustness_summary': robustness_summary,
                'most_robust_metric': max(robustness_summary.keys(), 
                                        key=lambda k: robustness_summary[k].get('mean_robustness', 0))
            }
            
        except Exception as e:
            logger.error(f"❌ Robustness analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _perform_metric_significance_tests(self, metric_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests between metrics."""
        try:
            from scipy.stats import ttest_ind
            
            metrics = list(metric_results.keys())
            significance_matrix = {}
            
            for i, metric1 in enumerate(metrics):
                significance_matrix[metric1] = {}
                for j, metric2 in enumerate(metrics):
                    if i != j and 'all_scores' in metric_results[metric1] and 'all_scores' in metric_results[metric2]:
                        scores1 = metric_results[metric1]['all_scores']
                        scores2 = metric_results[metric2]['all_scores']
                        
                        # Perform t-test
                        statistic, p_value = ttest_ind(scores1, scores2)
                        
                        significance_matrix[metric1][metric2] = {
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size': (np.mean(scores1) - np.mean(scores2)) / np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                        }
                    else:
                        significance_matrix[metric1][metric2] = {'error': 'Insufficient data'}
            
            return significance_matrix
            
        except Exception as e:
            logger.error(f"❌ Significance testing failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_scaling_impact(self, scaling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate impact of scaling on each metric."""
        try:
            impact_scores = {}
            
            for metric in ['euclidean', 'manhattan', 'chebyshev']:
                scores = []
                scalers = []
                
                for scaler_name, scaler_results in scaling_results.items():
                    if metric in scaler_results and 'mean_score' in scaler_results[metric]:
                        scores.append(scaler_results[metric]['mean_score'])
                        scalers.append(scaler_name)
                
                if len(scores) > 1:
                    impact_scores[metric] = {
                        'score_range': max(scores) - min(scores),
                        'best_scaling': scalers[np.argmax(scores)],
                        'worst_scaling': scalers[np.argmin(scores)],
                        'scaling_sensitivity': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
                    }
                else:
                    impact_scores[metric] = {'error': 'Insufficient scaling results'}
            
            return impact_scores
            
        except Exception as e:
            logger.error(f"❌ Scaling impact calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def _find_optimal_dimensions(self, dimensionality_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find optimal number of dimensions for each metric."""
        try:
            optimal_dims = {}
            
            for metric, results in dimensionality_results.items():
                pca_scores = np.array(results.get('pca_scores', []))
                selection_scores = np.array(results.get('selection_scores', []))
                dimensions = results.get('dimensions', [])
                
                # Find optimal for PCA
                if len(pca_scores) > 0 and not np.all(np.isnan(pca_scores)):
                    valid_pca = pca_scores[~np.isnan(pca_scores)]
                    valid_dims_pca = np.array(dimensions)[~np.isnan(pca_scores)]
                    optimal_pca_idx = np.argmax(valid_pca)
                    optimal_pca_dim = valid_dims_pca[optimal_pca_idx]
                else:
                    optimal_pca_dim = None
                
                # Find optimal for feature selection
                if len(selection_scores) > 0 and not np.all(np.isnan(selection_scores)):
                    valid_sel = selection_scores[~np.isnan(selection_scores)]
                    valid_dims_sel = np.array(dimensions)[~np.isnan(selection_scores)]
                    optimal_sel_idx = np.argmax(valid_sel)
                    optimal_sel_dim = valid_dims_sel[optimal_sel_idx]
                else:
                    optimal_sel_dim = None
                
                optimal_dims[metric] = {
                    'optimal_pca_dimensions': optimal_pca_dim,
                    'optimal_selection_dimensions': optimal_sel_dim,
                    'pca_score_at_optimal': valid_pca[optimal_pca_idx] if optimal_pca_dim is not None else None,
                    'selection_score_at_optimal': valid_sel[optimal_sel_idx] if optimal_sel_dim is not None else None
                }
            
            return optimal_dims
            
        except Exception as e:
            logger.error(f"❌ Optimal dimensions finding failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_metric_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metric recommendations."""
        try:
            recommendations = []
            
            # From basic comparison
            basic_comp = analysis_results.get('basic_comparison', {})
            if 'best_metric' in basic_comp:
                recommendations.append({
                    'type': 'performance_optimal',
                    'metric': basic_comp['best_metric'],
                    'score': basic_comp.get('best_score', 0),
                    'confidence': basic_comp.get('recommendation', {}).get('confidence', 'medium'),
                    'reason': 'Highest cross-validation score'
                })
            
            # From robustness analysis
            robustness = analysis_results.get('robustness_analysis', {})
            if 'most_robust_metric' in robustness:
                recommendations.append({
                    'type': 'robustness_optimal',
                    'metric': robustness['most_robust_metric'],
                    'reason': 'Most robust to noise and outliers'
                })
            
            # From scaling analysis
            scaling = analysis_results.get('feature_scaling_impact', {})
            if 'best_scalers' in scaling:
                for metric, scaler_info in scaling['best_scalers'].items():
                    if scaler_info['best_score'] > 0.1:  # Arbitrary threshold
                        recommendations.append({
                            'type': 'scaling_dependent',
                            'metric': metric,
                            'best_scaler': scaler_info['best_scaler'],
                            'score': scaler_info['best_score'],
                            'reason': f'Best with {scaler_info["best_scaler"]} scaling'
                        })
            
            # Generate final recommendation
            if recommendations:
                # Prefer performance optimal if available and confident
                performance_recs = [r for r in recommendations if r['type'] == 'performance_optimal']
                if performance_recs and performance_recs[0].get('confidence') in ['high', 'medium']:
                    final_rec = performance_recs[0]
                else:
                    # Use most common metric
                    metrics = [r['metric'] for r in recommendations]
                    from collections import Counter
                    most_common = Counter(metrics).most_common(1)
                    final_rec = {
                        'type': 'consensus',
                        'metric': most_common[0][0] if most_common else 'euclidean',
                        'reason': f'Most recommended across {len(recommendations)} analyses'
                    }
            else:
                final_rec = {
                    'type': 'default',
                    'metric': 'euclidean',
                    'reason': 'Default choice (no analysis available)'
                }
            
            return {
                'all_recommendations': recommendations,
                'final_recommendation': final_rec,
                'analysis_methods_used': len([k for k in analysis_results.keys() if 'error' not in analysis_results[k]])
            }
            
        except Exception as e:
            logger.error(f"❌ Metric recommendations generation failed: {str(e)}")
            return {'error': str(e)}
    
    # ==================== ALGORITHM PERFORMANCE ANALYSIS ====================
    
    def analyze_algorithm_efficiency(self) -> Dict[str, Any]:
        """
        Comprehensive algorithm efficiency analysis.
        
        Returns:
        --------
        Dict[str, Any]
            Algorithm efficiency analysis results
        """
        try:
            if not self.core.is_fitted:
                return {'error': 'Core model must be fitted before analysis'}
            
            cache_key = 'algorithm_efficiency'
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
            
            logger.info("🔍 Starting algorithm efficiency analysis...")
            
            X, y = self.core.X_train_scaled_, self.core.y_train_
            
            # Core algorithm analysis from core component
            basic_analysis = self.core.analyze_algorithm_performance(self.core.X_train_, self.core.y_train_)
            
            # Extended analysis
            results = {
                'basic_performance': basic_analysis,
                'scalability_analysis': self._algorithm_scalability_analysis(X, y),
                'memory_analysis': self._algorithm_memory_analysis(X, y),
                'dimensionality_impact': self._algorithm_dimensionality_impact(X, y),
                'parameter_sensitivity': self._algorithm_parameter_sensitivity(X, y)
            }
            
            # Generate recommendations
            results['recommendations'] = self._generate_algorithm_recommendations(results)
            
            # Cache results
            self._analysis_cache[cache_key] = results
            
            logger.info("✅ Algorithm efficiency analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Algorithm efficiency analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _algorithm_scalability_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze algorithm scalability with increasing data size."""
        try:
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            sample_sizes = [min(len(X), size) for size in [100, 250, 500, 1000, len(X)]]
            sample_sizes = list(set(sample_sizes))  # Remove duplicates
            sample_sizes.sort()
            
            results = {}
            
            for algo in algorithms:
                results[algo] = {
                    'sample_sizes': sample_sizes,
                    'training_times': [],
                    'prediction_times': [],
                    'memory_usage': [],
                    'scalability_score': 0
                }
                
                for size in sample_sizes:
                    try:
                        # Subsample data
                        indices = np.random.choice(len(X), size, replace=False)
                        X_sub, y_sub = X[indices], y[indices]
                        
                        # Create model
                        temp_model = KNeighborsRegressor(
                            n_neighbors=min(self.core.n_neighbors, size - 1),
                            weights=self.core.weights,
                            algorithm=algo,
                            metric=self.core.metric
                        )
                        
                        # Training time
                        start_time = time.time()
                        temp_model.fit(X_sub, y_sub)
                        train_time = time.time() - start_time
                        results[algo]['training_times'].append(train_time)
                        
                        # Prediction time
                        test_size = min(50, size // 4)
                        X_test = X_sub[:test_size]
                        start_time = time.time()
                        temp_model.predict(X_test)
                        pred_time = time.time() - start_time
                        results[algo]['prediction_times'].append(pred_time)
                        
                        # Memory estimation
                        from .knn_core import estimate_memory_usage
                        memory_info = estimate_memory_usage(size, X.shape[1], algo)
                        results[algo]['memory_usage'].append(memory_info['estimated_memory_mb'])
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Scalability test failed for {algo}, size={size}: {str(e)}")
                        results[algo]['training_times'].append(np.nan)
                        results[algo]['prediction_times'].append(np.nan)
                        results[algo]['memory_usage'].append(np.nan)
                
                # Calculate scalability score (lower is better)
                train_times = np.array(results[algo]['training_times'])
                valid_times = train_times[~np.isnan(train_times)]
                valid_sizes = np.array(sample_sizes)[~np.isnan(train_times)]
                
                if len(valid_times) > 1:
                    # Fit polynomial to estimate complexity
                    coeffs = np.polyfit(np.log(valid_sizes), np.log(valid_times), 1)
                    complexity_factor = coeffs[0]  # Slope in log-log space
                    results[algo]['scalability_score'] = complexity_factor
                else:
                    results[algo]['scalability_score'] = float('inf')
            
            # Find most scalable algorithm
            scalability_scores = {algo: results[algo]['scalability_score'] 
                                for algo in algorithms if results[algo]['scalability_score'] < float('inf')}
            
            if scalability_scores:
                most_scalable = min(scalability_scores.keys(), key=lambda k: scalability_scores[k])
            else:
                most_scalable = 'auto'
            
            return {
                'algorithm_results': results,
                'most_scalable_algorithm': most_scalable,
                'scalability_ranking': sorted(scalability_scores.keys(), key=lambda k: scalability_scores[k])
            }
            
        except Exception as e:
            logger.error(f"❌ Scalability analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _algorithm_memory_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze memory usage of different algorithms."""
        try:
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            results = {}
            
            for algo in algorithms:
                try:
                    # Create model
                    temp_model = KNeighborsRegressor(
                        n_neighbors=self.core.n_neighbors,
                        weights=self.core.weights,
                        algorithm=algo,
                        metric=self.core.metric
                    )
                    
                    # Estimate memory usage
                    from .knn_core import estimate_memory_usage
                    memory_info = estimate_memory_usage(len(X), X.shape[1], algo)
                    
                    # Training memory (rough estimation)
                    start_memory = self._get_memory_usage()
                    temp_model.fit(X, y)
                    end_memory = self._get_memory_usage()
                    
                    actual_memory_mb = max(0, end_memory - start_memory) if start_memory is not None and end_memory is not None else None
                    
                    results[algo] = {
                        'estimated_memory_mb': memory_info['estimated_memory_mb'],
                        'base_data_mb': memory_info['base_data_mb'],
                        'overhead_factor': memory_info['overhead_factor'],
                        'actual_memory_mb': actual_memory_mb,
                        'memory_efficiency_score': memory_info['base_data_mb'] / memory_info['estimated_memory_mb']
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ Memory analysis failed for {algo}: {str(e)}")
                    results[algo] = {'error': str(e)}
            
            # Find most memory efficient
            efficiency_scores = {algo: results[algo].get('memory_efficiency_score', 0) 
                               for algo in algorithms if 'error' not in results[algo]}
            
            if efficiency_scores:
                most_efficient = max(efficiency_scores.keys(), key=lambda k: efficiency_scores[k])
            else:
                most_efficient = 'auto'
            
            return {
                'algorithm_memory_results': results,
                'most_memory_efficient': most_efficient,
                'efficiency_ranking': sorted(efficiency_scores.keys(), key=lambda k: efficiency_scores[k], reverse=True)
            }
            
        except Exception as e:
            logger.error(f"❌ Memory analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return None
        except Exception:
            return None
    
    def _algorithm_dimensionality_impact(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze how dimensionality affects different algorithms."""
        try:
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            
            # Test with different feature subsets
            n_features = X.shape[1]
            if n_features > 10:
                feature_counts = [5, 10, n_features // 2, n_features]
            else:
                feature_counts = list(range(2, n_features + 1))
            
            results = {}
            
            for algo in algorithms:
                results[algo] = {
                    'feature_counts': feature_counts,
                    'training_times': [],
                    'prediction_times': [],
                    'performance_scores': []
                }
                
                for n_feat in feature_counts:
                    try:
                        # Select features randomly
                        feat_indices = np.random.choice(n_features, n_feat, replace=False)
                        X_subset = X[:, feat_indices]
                        
                        # Create and train model
                        temp_model = KNeighborsRegressor(
                            n_neighbors=self.core.n_neighbors,
                            weights=self.core.weights,
                            algorithm=algo,
                            metric=self.core.metric
                        )
                        
                        # Training time
                        start_time = time.time()
                        temp_model.fit(X_subset, y)
                        train_time = time.time() - start_time
                        results[algo]['training_times'].append(train_time)
                        
                        # Prediction time
                        start_time = time.time()
                        temp_model.predict(X_subset[:min(50, len(X_subset))])
                        pred_time = time.time() - start_time
                        results[algo]['prediction_times'].append(pred_time)
                        
                        # Performance
                        cv_score = cross_val_score(temp_model, X_subset, y, cv=3, scoring='r2').mean()
                        results[algo]['performance_scores'].append(cv_score)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Dimensionality test failed for {algo}, features={n_feat}: {str(e)}")
                        results[algo]['training_times'].append(np.nan)
                        results[algo]['prediction_times'].append(np.nan)
                        results[algo]['performance_scores'].append(np.nan)
                
                # Calculate curse of dimensionality factor
                train_times = np.array(results[algo]['training_times'])
                valid_times = train_times[~np.isnan(train_times)]
                valid_features = np.array(feature_counts)[~np.isnan(train_times)]
                
                if len(valid_times) > 1 and len(valid_features) > 1:
                    # Linear regression to estimate impact of dimensionality
                    coeffs = np.polyfit(valid_features, valid_times, 1)
                    dimensionality_impact = coeffs[0]  # Slope
                else:
                    dimensionality_impact = 0
                
                results[algo]['dimensionality_impact'] = dimensionality_impact
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Dimensionality impact analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _algorithm_parameter_sensitivity(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze algorithm sensitivity to parameter changes."""
        try:
            algorithms = ['ball_tree', 'kd_tree']  # Parameters mainly affect tree algorithms
            results = {}
            
            # Test different leaf sizes for tree algorithms
            leaf_sizes = [10, 20, 30, 50]
            
            for algo in algorithms:
                results[algo] = {
                    'leaf_sizes': leaf_sizes,
                    'training_times': [],
                    'prediction_times': [],
                    'performance_scores': []
                }
                
                for leaf_size in leaf_sizes:
                    try:
                        temp_model = KNeighborsRegressor(
                            n_neighbors=self.core.n_neighbors,
                            weights=self.core.weights,
                            algorithm=algo,
                            metric=self.core.metric,
                            leaf_size=leaf_size
                        )
                        
                        # Training time
                        start_time = time.time()
                        temp_model.fit(X, y)
                        train_time = time.time() - start_time
                        results[algo]['training_times'].append(train_time)
                        
                        # Prediction time
                        start_time = time.time()
                        temp_model.predict(X[:min(100, len(X))])
                        pred_time = time.time() - start_time
                        results[algo]['prediction_times'].append(pred_time)
                        
                        # Performance
                        cv_score = cross_val_score(temp_model, X, y, cv=3, scoring='r2').mean()
                        results[algo]['performance_scores'].append(cv_score)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Parameter sensitivity test failed for {algo}, leaf_size={leaf_size}: {str(e)}")
                        results[algo]['training_times'].append(np.nan)
                        results[algo]['prediction_times'].append(np.nan)
                        results[algo]['performance_scores'].append(np.nan)
                
                # Calculate sensitivity score (coefficient of variation)
                train_times = np.array(results[algo]['training_times'])
                valid_times = train_times[~np.isnan(train_times)]
                
                if len(valid_times) > 1:
                    sensitivity_score = np.std(valid_times) / np.mean(valid_times)
                else:
                    sensitivity_score = 0
                
                results[algo]['parameter_sensitivity'] = sensitivity_score
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Parameter sensitivity analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_algorithm_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive algorithm recommendations."""
        try:
            recommendations = []
            
            # From basic performance
            basic_perf = analysis_results.get('basic_performance', {})
            if 'fastest_training' in basic_perf:
                recommendations.append({
                    'type': 'speed_optimal',
                    'algorithm': basic_perf['fastest_training'],
                    'reason': 'Fastest training time'
                })
            
            if 'fastest_prediction' in basic_perf:
                recommendations.append({
                    'type': 'prediction_speed_optimal',
                    'algorithm': basic_perf['fastest_prediction'],
                    'reason': 'Fastest prediction time'
                })
            
            # From scalability
            scalability = analysis_results.get('scalability_analysis', {})
            if 'most_scalable_algorithm' in scalability:
                recommendations.append({
                    'type': 'scalability_optimal',
                    'algorithm': scalability['most_scalable_algorithm'],
                    'reason': 'Best scalability with data size'
                })
            
            # From memory analysis
            memory = analysis_results.get('memory_analysis', {})
            if 'most_memory_efficient' in memory:
                recommendations.append({
                    'type': 'memory_optimal',
                    'algorithm': memory['most_memory_efficient'],
                    'reason': 'Most memory efficient'
                })
            
            # Generate final recommendation based on context
            data_size = len(self.core.X_train_) if self.core.X_train_ is not None else 0
            n_features = self.core.X_train_.shape[1] if self.core.X_train_ is not None else 0
            
            if data_size < 1000 and n_features < 20:
                context_rec = 'brute'
                context_reason = 'Small dataset: brute force is simple and effective'
            elif n_features > 20:
                context_rec = 'auto'
                context_reason = 'High dimensionality: let sklearn choose the best algorithm'
            elif data_size > 10000:
                context_rec = 'ball_tree'
                context_reason = 'Large dataset: ball_tree typically scales better'
            else:
                context_rec = 'auto'
                context_reason = 'Medium-sized dataset: auto selection is recommended'
            
            recommendations.append({
                'type': 'context_based',
                'algorithm': context_rec,
                'reason': context_reason
            })
            
            # Final recommendation (prefer context-based or auto)
            final_rec = {
                'algorithm': context_rec,
                'reason': context_reason,
                'confidence': 'high'
            }
            
            return {
                'all_recommendations': recommendations,
                'final_recommendation': final_rec,
                'data_context': {
                    'n_samples': data_size,
                    'n_features': n_features,
                    'size_category': 'small' if data_size < 1000 else 'medium' if data_size < 10000 else 'large'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Algorithm recommendations generation failed: {str(e)}")
            return {'error': str(e)}
    
    # ==================== CROSS-VALIDATION ANALYSIS ====================
    
    def analyze_cross_validation(self) -> Dict[str, Any]:
        """
        Comprehensive cross-validation analysis.
        
        Returns:
        --------
        Dict[str, Any]
            Cross-validation analysis results
        """
        try:
            if not self.core.is_fitted:
                return {'error': 'Core model must be fitted before analysis'}
            
            cache_key = 'cross_validation'
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
            
            logger.info("🔍 Starting cross-validation analysis...")
            
            X, y = self.core.X_train_scaled_, self.core.y_train_
            
            results = {
                'basic_cv': self._basic_cv_analysis(X, y),
                'stratified_cv': self._stratified_cv_analysis(X, y),
                'time_series_cv': self._time_series_cv_analysis(X, y),
                'stability_analysis': self._cv_stability_analysis(X, y),
                'learning_curves': self._learning_curve_analysis(X, y),
                'scoring_metrics_comparison': self._scoring_metrics_analysis(X, y)
            }
            
            # Generate CV recommendations
            results['recommendations'] = self._generate_cv_recommendations(results)
            
            # Cache results
            self._analysis_cache[cache_key] = results
            
            logger.info("✅ Cross-validation analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Cross-validation analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _basic_cv_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Basic cross-validation analysis with multiple metrics."""
        try:
            cv_folds = self.config.get('cv_folds', 5)
            
            # Multiple scoring metrics
            scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            results = {}
            
            for metric in scoring_metrics:
                cv_scores = cross_val_score(
                    self.core.model, X, y,
                    cv=cv_folds,
                    scoring=metric,
                    n_jobs=self.config.get('n_jobs', 1)
                )
                
                # Convert negative scores to positive for MSE and MAE
                if metric.startswith('neg_'):
                    cv_scores = -cv_scores
                    metric_name = metric[4:]  # Remove 'neg_' prefix
                else:
                    metric_name = metric
                
                results[metric_name] = {
                    'scores': cv_scores.tolist(),
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'min': cv_scores.min(),
                    'max': cv_scores.max(),
                    'cv_folds': cv_folds
                }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Basic CV analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _stratified_cv_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Stratified cross-validation for regression (binned targets)."""
        try:
            from sklearn.model_selection import StratifiedKFold
            
            # Bin targets for stratification
            n_bins = min(10, len(np.unique(y)))
            y_binned = pd.cut(y, bins=n_bins, labels=False)
            
            # Remove any NaN bins
            valid_indices = ~pd.isna(y_binned)
            X_valid = X[valid_indices]
            y_valid = y[valid_indices]
            y_binned_valid = y_binned[valid_indices]
            
            if len(np.unique(y_binned_valid)) < 2:
                return {'error': 'Insufficient variation in target for stratification'}
            
            skf = StratifiedKFold(n_splits=min(5, len(np.unique(y_binned_valid))), shuffle=True, random_state=42)
            
            stratified_scores = []
            fold_distributions = []
            
            for train_idx, val_idx in skf.split(X_valid, y_binned_valid):
                X_train_fold, X_val_fold = X_valid[train_idx], X_valid[val_idx]
                y_train_fold, y_val_fold = y_valid[train_idx], y_valid[val_idx]
                
                # Train temporary model
                temp_model = KNeighborsRegressor(
                    n_neighbors=self.core.n_neighbors,
                    weights=self.core.weights,
                    algorithm=self.core.algorithm,
                    metric=self.core.metric
                )
                
                temp_model.fit(X_train_fold, y_train_fold)
                pred = temp_model.predict(X_val_fold)
                score = r2_score(y_val_fold, pred)
                stratified_scores.append(score)
                
                # Track target distribution in each fold
                fold_distributions.append({
                    'mean': y_val_fold.mean(),
                    'std': y_val_fold.std(),
                    'min': y_val_fold.min(),
                    'max': y_val_fold.max()
                })
            
            return {
                'stratified_scores': stratified_scores,
                'mean_score': np.mean(stratified_scores),
                'std_score': np.std(stratified_scores),
                'fold_distributions': fold_distributions,
                'distribution_consistency': np.std([f['mean'] for f in fold_distributions])
            }
            
        except Exception as e:
            logger.error(f"❌ Stratified CV analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _time_series_cv_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Time series cross-validation (assuming temporal order)."""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            
            n_splits = min(5, len(X) // 3)
            if n_splits < 2:
                return {'error': 'Insufficient data for time series CV'}
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            ts_scores = []
            train_sizes = []
            val_sizes = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                if len(X_train_fold) < self.core.n_neighbors:
                    continue
                
                # Train temporary model
                temp_model = KNeighborsRegressor(
                    n_neighbors=min(self.core.n_neighbors, len(X_train_fold) - 1),
                    weights=self.core.weights,
                    algorithm=self.core.algorithm,
                    metric=self.core.metric
                )
                
                temp_model.fit(X_train_fold, y_train_fold)
                pred = temp_model.predict(X_val_fold)
                score = r2_score(y_val_fold, pred)
                ts_scores.append(score)
                
                train_sizes.append(len(X_train_fold))
                val_sizes.append(len(X_val_fold))
            
            if not ts_scores:
                return {'error': 'No valid time series CV folds'}
            
            # Analyze trend in scores over time
            score_trend = np.polyfit(range(len(ts_scores)), ts_scores, 1)[0] if len(ts_scores) > 1 else 0
            
            return {
                'ts_scores': ts_scores,
                'mean_score': np.mean(ts_scores),
                'std_score': np.std(ts_scores),
                'score_trend': score_trend,
                'train_sizes': train_sizes,
                'val_sizes': val_sizes,
                'temporal_stability': 'stable' if abs(score_trend) < 0.01 else 'declining' if score_trend < 0 else 'improving'
            }
            
        except Exception as e:
            logger.error(f"❌ Time series CV analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _cv_stability_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze stability of CV results across multiple runs."""
        try:
            n_runs = 10
            cv_folds = self.config.get('cv_folds', 5)
            
            all_run_scores = []
            all_run_means = []
            all_run_stds = []
            
            for run in range(n_runs):
                cv_scores = cross_val_score(
                    self.core.model, X, y,
                    cv=cv_folds,
                    scoring='r2',
                    random_state=run
                )
                
                all_run_scores.extend(cv_scores.tolist())
                all_run_means.append(cv_scores.mean())
                all_run_stds.append(cv_scores.std())
            
            # Calculate stability metrics
            mean_of_means = np.mean(all_run_means)
            std_of_means = np.std(all_run_means)
            mean_of_stds = np.mean(all_run_stds)
            
            # Coefficient of variation as stability measure
            stability_score = std_of_means / mean_of_means if mean_of_means > 0 else float('inf')
            
            return {
                'all_scores': all_run_scores,
                'run_means': all_run_means,
                'run_stds': all_run_stds,
                'overall_mean': mean_of_means,
                'overall_std': np.std(all_run_scores),
                'stability_score': stability_score,
                'consistency_rating': 'high' if stability_score < 0.05 else 'medium' if stability_score < 0.1 else 'low',
                'mean_variability': std_of_means,
                'fold_variability': mean_of_stds
            }
            
        except Exception as e:
            logger.error(f"❌ CV stability analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _learning_curve_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze learning curves to understand training data requirements."""
        try:
            # Define training sizes
            max_size = len(X)
            if max_size > 1000:
                train_sizes = np.linspace(0.1, 1.0, 10)
            else:
                train_sizes = np.linspace(0.2, 1.0, min(8, max_size // 20))
            
            # Generate learning curves
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.core.model, X, y,
                train_sizes=train_sizes,
                cv=min(3, len(X) // 5),
                scoring='r2',
                n_jobs=self.config.get('n_jobs', 1),
                random_state=42
            )
            
            # Calculate means and stds
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            val_scores_mean = np.mean(val_scores, axis=1)
            val_scores_std = np.std(val_scores, axis=1)
            
            # Analyze convergence
            if len(val_scores_mean) > 3:
                # Check if validation score is still improving
                recent_improvement = val_scores_mean[-1] - val_scores_mean[-3]
                convergence_status = 'converged' if recent_improvement < 0.01 else 'improving'
                
                # Estimate optimal training size
                optimal_size_idx = np.argmax(val_scores_mean)
                optimal_size = train_sizes_abs[optimal_size_idx]
            else:
                convergence_status = 'unknown'
                optimal_size = max_size
            
            # Bias-variance analysis from learning curves
            final_gap = train_scores_mean[-1] - val_scores_mean[-1]
            bias_indicator = 1 - val_scores_mean[-1]  # How far from perfect score
            variance_indicator = final_gap  # Gap between train and validation
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores_mean.tolist(),
                'train_scores_std': train_scores_std.tolist(),
                'val_scores_mean': val_scores_mean.tolist(),
                'val_scores_std': val_scores_std.tolist(),
                'convergence_status': convergence_status,
                'optimal_training_size': optimal_size,
                'final_train_score': train_scores_mean[-1],
                'final_val_score': val_scores_mean[-1],
                'overfitting_gap': final_gap,
                'bias_indicator': bias_indicator,
                'variance_indicator': variance_indicator,
                'learning_efficiency': val_scores_mean[-1] / (train_sizes_abs[-1] / max_size)
            }
            
        except Exception as e:
            logger.error(f"❌ Learning curve analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _scoring_metrics_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compare different scoring metrics for cross-validation."""
        try:
            scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
            results = {}
            correlations = {}
            
            cv_folds = self.config.get('cv_folds', 5)
            
            # Calculate scores for each metric
            all_scores = {}
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(
                        self.core.model, X, y,
                        cv=cv_folds,
                        scoring=metric
                    )
                    
                    # Convert negative scores to positive for interpretability
                    if metric.startswith('neg_'):
                        scores = -scores
                        metric_name = metric[4:]
                    else:
                        metric_name = metric
                    
                    results[metric_name] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'coefficient_of_variation': scores.std() / abs(scores.mean()) if scores.mean() != 0 else float('inf')
                    }
                    
                    all_scores[metric_name] = scores
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to calculate {metric}: {str(e)}")
                    results[metric] = {'error': str(e)}
            
            # Calculate correlations between metrics
            metric_names = list(all_scores.keys())
            for i, metric1 in enumerate(metric_names):
                correlations[metric1] = {}
                for j, metric2 in enumerate(metric_names):
                    if i != j:
                        corr = np.corrcoef(all_scores[metric1], all_scores[metric2])[0, 1]
                        correlations[metric1][metric2] = corr
            
            # Identify most stable metric
            cv_values = {name: data.get('coefficient_of_variation', float('inf')) 
                        for name, data in results.items() if 'error' not in data}
            
            if cv_values:
                most_stable_metric = min(cv_values.keys(), key=lambda k: cv_values[k])
            else:
                most_stable_metric = 'r2'
            
            return {
                'metric_results': results,
                'metric_correlations': correlations,
                'most_stable_metric': most_stable_metric,
                'stability_ranking': sorted(cv_values.keys(), key=lambda k: cv_values[k])
            }
            
        except Exception as e:
            logger.error(f"❌ Scoring metrics analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_cv_recommendations(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on CV analysis."""
        try:
            recommendations = []
            
            # From basic CV
            basic_cv = cv_results.get('basic_cv', {})
            if 'r2' in basic_cv:
                r2_std = basic_cv['r2'].get('std', 0)
                if r2_std < 0.05:
                    recommendations.append({
                        'type': 'cv_stability',
                        'message': 'Model shows stable cross-validation performance',
                        'confidence': 'high'
                    })
                elif r2_std > 0.15:
                    recommendations.append({
                        'type': 'cv_instability',
                        'message': 'High CV variance suggests model instability - consider parameter tuning',
                        'confidence': 'medium'
                    })
            
            # From learning curves
            learning_curves = cv_results.get('learning_curves', {})
            if 'convergence_status' in learning_curves:
                if learning_curves['convergence_status'] == 'improving':
                    recommendations.append({
                        'type': 'data_need',
                        'message': 'Model could benefit from more training data',
                        'confidence': 'medium'
                    })
                
                overfitting_gap = learning_curves.get('overfitting_gap', 0)
                if overfitting_gap > 0.1:
                    recommendations.append({
                        'type': 'overfitting',
                        'message': 'Significant overfitting detected - consider increasing K or regularization',
                        'confidence': 'high'
                    })
            
            # From stability analysis
            stability = cv_results.get('stability_analysis', {})
            if 'consistency_rating' in stability:
                rating = stability['consistency_rating']
                if rating == 'low':
                    recommendations.append({
                        'type': 'consistency',
                        'message': 'Low consistency across CV runs - model may be sensitive to data splits',
                        'confidence': 'high'
                    })
            
            return {
                'recommendations': recommendations,
                'overall_cv_quality': self._assess_cv_quality(cv_results)
            }
            
        except Exception as e:
            logger.error(f"❌ CV recommendations generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _assess_cv_quality(self, cv_results: Dict[str, Any]) -> str:
        """Assess overall quality of cross-validation results."""
        try:
            quality_indicators = []
            
            # Check basic CV score
            basic_cv = cv_results.get('basic_cv', {})
            if 'r2' in basic_cv:
                r2_mean = basic_cv['r2'].get('mean', 0)
                if r2_mean > 0.8:
                    quality_indicators.append('excellent_performance')
                elif r2_mean > 0.6:
                    quality_indicators.append('good_performance')
                elif r2_mean > 0.3:
                    quality_indicators.append('moderate_performance')
                else:
                    quality_indicators.append('poor_performance')
            
            # Check stability
            stability = cv_results.get('stability_analysis', {})
            if 'consistency_rating' in stability:
                rating = stability['consistency_rating']
                if rating == 'high':
                    quality_indicators.append('stable')
                elif rating == 'low':
                    quality_indicators.append('unstable')
            
            # Overall assessment
            if 'excellent_performance' in quality_indicators and 'stable' in quality_indicators:
                return 'excellent'
            elif 'good_performance' in quality_indicators and 'unstable' not in quality_indicators:
                return 'good'
            elif 'poor_performance' in quality_indicators or 'unstable' in quality_indicators:
                return 'poor'
            else:
                return 'moderate'
                
        except Exception as e:
            return 'unknown'
    
    # ==================== FEATURE IMPORTANCE ANALYSIS ====================
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Comprehensive feature importance analysis for KNN.
        
        Returns:
        --------
        Dict[str, Any]
            Feature importance analysis results
        """
        try:
            if not self.core.is_fitted:
                return {'error': 'Core model must be fitted before analysis'}
            
            cache_key = 'feature_importance'
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
            
            logger.info("🔍 Starting feature importance analysis...")
            
            X, y = self.core.X_train_scaled_, self.core.y_train_
            
            results = {
                'permutation_importance': self._permutation_importance_analysis(X, y),
                'drop_column_importance': self._drop_column_importance_analysis(X, y),
                'distance_based_importance': self._distance_based_importance_analysis(X, y),
                'neighbor_consistency_importance': self._neighbor_consistency_importance(X, y),
                'local_importance_variation': self._local_importance_variation_analysis(X, y)
            }
            
            # Combine and rank features
            results['combined_ranking'] = self._combine_importance_rankings(results)
            results['feature_recommendations'] = self._generate_feature_recommendations(results)
            
            # Cache results
            self._analysis_cache[cache_key] = results
            
            logger.info("✅ Feature importance analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Feature importance analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _permutation_importance_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Standard permutation importance analysis."""
        try:
            # Permutation importance
            perm_importance = permutation_importance(
                self.core.model, X, y,
                n_repeats=self.config.get('n_repeats', 5),
                random_state=self.config.get('random_state', 42),
                scoring='r2',
                n_jobs=self.config.get('n_jobs', 1)
            )
            
            # Create feature importance dataframe
            n_features = X.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std,
                'importance_min': perm_importance.importances.min(axis=1),
                'importance_max': perm_importance.importances.max(axis=1)
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance_mean', ascending=False)
            
            # Calculate significance (features with importance > 2*std above mean)
            mean_importance = importance_df['importance_mean'].mean()
            std_importance = importance_df['importance_mean'].std()
            significance_threshold = mean_importance + 2 * std_importance
            
            significant_features = importance_df[
                importance_df['importance_mean'] > significance_threshold
            ]['feature'].tolist()
            
            return {
                'importance_scores': importance_df.to_dict('records'),
                'feature_ranking': importance_df['feature'].tolist(),
                'top_features': importance_df.head(5)['feature'].tolist(),
                'significant_features': significant_features,
                'importance_distribution': {
                    'mean': mean_importance,
                    'std': std_importance,
                    'min': importance_df['importance_mean'].min(),
                    'max': importance_df['importance_mean'].max()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Permutation importance analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _drop_column_importance_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Drop column importance analysis."""
        try:
            n_features = X.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # Baseline score with all features
            baseline_score = cross_val_score(
                self.core.model, X, y, cv=3, scoring='r2'
            ).mean()
            
            drop_importances = []
            
            for i in range(n_features):
                # Create dataset without feature i
                X_dropped = np.delete(X, i, axis=1)
                
                # Create temporary model (might need different n_neighbors if too close to n_features)
                temp_n_neighbors = min(self.core.n_neighbors, X_dropped.shape[1], len(X) - 1)
                temp_model = KNeighborsRegressor(
                    n_neighbors=temp_n_neighbors,
                    weights=self.core.weights,
                    algorithm=self.core.algorithm,
                    metric=self.core.metric
                )
                
                # Score without this feature
                dropped_score = cross_val_score(
                    temp_model, X_dropped, y, cv=3, scoring='r2'
                ).mean()
                
                # Importance is the decrease in performance
                importance = baseline_score - dropped_score
                drop_importances.append({
                    'feature': feature_names[i],
                    'baseline_score': baseline_score,
                    'dropped_score': dropped_score,
                    'importance': importance,
                    'relative_importance': importance / baseline_score if baseline_score > 0 else 0
                })
            
            # Sort by importance
            drop_importances.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'drop_importances': drop_importances,
                'feature_ranking': [item['feature'] for item in drop_importances],
                'baseline_score': baseline_score,
                'most_important_feature': drop_importances[0]['feature'] if drop_importances else None,
                'importance_range': {
                    'max': max(item['importance'] for item in drop_importances),
                    'min': min(item['importance'] for item in drop_importances),
                    'mean': np.mean([item['importance'] for item in drop_importances])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Drop column importance analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _distance_based_importance_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """KNN-specific distance-based feature importance."""
        try:
            n_features = X.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # Calculate feature-wise distance contributions
            distance_contributions = []
            
            # Sample a subset of points for efficiency
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_indices]
            
            for i in range(n_features):
                # Calculate distances using only feature i
                feature_distances = []
                
                for j in range(len(X_sample)):
                    # Get neighbors using current model
                    neighbors_info = self.core.get_neighbors_info(X_sample[j:j+1])
                    if 'neighbor_indices' in neighbors_info:
                        neighbor_indices = neighbors_info['neighbor_indices'][0]
                        
                        # Calculate contribution of feature i to total distances
                        point_feature_val = X_sample[j, i]
                        neighbor_feature_vals = X[neighbor_indices, i]
                        
                        # Feature-specific distance (absolute difference for simplicity)
                        feature_dist = np.mean(np.abs(point_feature_val - neighbor_feature_vals))
                        feature_distances.append(feature_dist)
                
                if feature_distances:
                    avg_feature_distance = np.mean(feature_distances)
                    distance_contributions.append({
                        'feature': feature_names[i],
                        'avg_distance_contribution': avg_feature_distance,
                        'distance_std': np.std(feature_distances),
                        'distance_range': max(feature_distances) - min(feature_distances)
                    })
                else:
                    distance_contributions.append({
                        'feature': feature_names[i],
                        'avg_distance_contribution': 0,
                        'distance_std': 0,
                        'distance_range': 0
                    })
            
            # Sort by distance contribution
            distance_contributions.sort(key=lambda x: x['avg_distance_contribution'], reverse=True)
            
            # Normalize contributions
            max_contrib = max(item['avg_distance_contribution'] for item in distance_contributions)
            if max_contrib > 0:
                for item in distance_contributions:
                    item['normalized_contribution'] = item['avg_distance_contribution'] / max_contrib
            
            return {
                'distance_contributions': distance_contributions,
                'feature_ranking': [item['feature'] for item in distance_contributions],
                'most_discriminative_feature': distance_contributions[0]['feature'] if distance_contributions else None
            }
            
        except Exception as e:
            logger.error(f"❌ Distance-based importance analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _neighbor_consistency_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance based on neighbor consistency."""
        try:
            n_features = X.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            consistency_scores = []
            
            # Sample points for efficiency
            sample_size = min(50, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            
            for i in range(n_features):
                feature_consistency = []
                
                for j in sample_indices:
                    # Get neighbors for this point
                    neighbors_info = self.core.get_neighbors_info(X[j:j+1])
                    if 'neighbor_indices' in neighbors_info:
                        neighbor_indices = neighbors_info['neighbor_indices'][0]
                        
                        # Calculate consistency of feature i among neighbors
                        point_feature_val = X[j, i]
                        neighbor_feature_vals = X[neighbor_indices, i]
                        neighbor_target_vals = y[neighbor_indices]
                        
                        # Consistency: correlation between feature similarity and target similarity
                        feature_diffs = np.abs(neighbor_feature_vals - point_feature_val)
                        target_diffs = np.abs(neighbor_target_vals - y[j])
                        
                        if len(feature_diffs) > 1 and np.std(feature_diffs) > 0 and np.std(target_diffs) > 0:
                            consistency = -np.corrcoef(feature_diffs, target_diffs)[0, 1]
                            # Negative correlation means similar features -> similar targets (good)
                            feature_consistency.append(consistency if not np.isnan(consistency) else 0)
                
                if feature_consistency:
                    avg_consistency = np.mean(feature_consistency)
                    consistency_scores.append({
                        'feature': feature_names[i],
                        'consistency_score': avg_consistency,
                        'consistency_std': np.std(feature_consistency),
                        'n_evaluations': len(feature_consistency)
                    })
                else:
                    consistency_scores.append({
                        'feature': feature_names[i],
                        'consistency_score': 0,
                        'consistency_std': 0,
                        'n_evaluations': 0
                    })
            
            # Sort by consistency
            consistency_scores.sort(key=lambda x: x['consistency_score'], reverse=True)
            
            return {
                'consistency_scores': consistency_scores,
                'feature_ranking': [item['feature'] for item in consistency_scores],
                'most_consistent_feature': consistency_scores[0]['feature'] if consistency_scores else None
            }
            
        except Exception as e:
            logger.error(f"❌ Neighbor consistency importance analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _local_importance_variation_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze how feature importance varies across different regions of the feature space."""
        try:
            from sklearn.cluster import KMeans
            
            n_features = X.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # Cluster the data into regions
            n_clusters = min(5, len(X) // 20)
            if n_clusters < 2:
                return {'error': 'Insufficient data for local importance analysis'}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            local_importances = {}
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_X = X[cluster_mask]
                cluster_y = y[cluster_mask]
                
                if len(cluster_X) < 3:  # Skip small clusters
                    continue
                
                # Calculate permutation importance for this cluster
                try:
                    temp_model = KNeighborsRegressor(
                        n_neighbors=min(self.core.n_neighbors, len(cluster_X) - 1),
                        weights=self.core.weights,
                        algorithm=self.core.algorithm,
                        metric=self.core.metric
                    )
                    
                    temp_model.fit(cluster_X, cluster_y)
                    
                    perm_importance = permutation_importance(
                        temp_model, cluster_X, cluster_y,
                        n_repeats=3,  # Fewer repeats for speed
                        random_state=42,
                        scoring='r2'
                    )
                    
                    local_importances[f'cluster_{cluster_id}'] = {
                        'feature_importances': dict(zip(feature_names, perm_importance.importances_mean)),
                        'cluster_size': len(cluster_X),
                        'cluster_center': kmeans.cluster_centers_[cluster_id].tolist()
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed local importance for cluster {cluster_id}: {str(e)}")
                    continue
            
            # Analyze variation across clusters
            importance_variations = {}
            for feature in feature_names:
                feature_importances = []
                for cluster_data in local_importances.values():
                    feature_importances.append(cluster_data['feature_importances'].get(feature, 0))
                
                if feature_importances:
                    importance_variations[feature] = {
                        'mean': np.mean(feature_importances),
                        'std': np.std(feature_importances),
                        'min': min(feature_importances),
                        'max': max(feature_importances),
                        'variation_coefficient': np.std(feature_importances) / (np.mean(feature_importances) + 1e-10)
                    }
            
            return {
                'local_importances': local_importances,
                'importance_variations': importance_variations,
                'n_clusters': n_clusters,
                'most_variable_feature': max(importance_variations.keys(), 
                                           key=lambda f: importance_variations[f]['variation_coefficient'])
                                         if importance_variations else None
            }
            
        except Exception as e:
            logger.error(f"❌ Local importance variation analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _combine_importance_rankings(self, importance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple importance rankings into a consensus ranking."""
        try:
            # Collect rankings from different methods
            rankings = {}
            
            # Permutation importance
            perm_imp = importance_results.get('permutation_importance', {})
            if 'feature_ranking' in perm_imp:
                rankings['permutation'] = perm_imp['feature_ranking']
            
            # Drop column importance
            drop_imp = importance_results.get('drop_column_importance', {})
            if 'feature_ranking' in drop_imp:
                rankings['drop_column'] = drop_imp['feature_ranking']
            
            # Distance-based importance
            dist_imp = importance_results.get('distance_based_importance', {})
            if 'feature_ranking' in dist_imp:
                rankings['distance_based'] = dist_imp['feature_ranking']
            
            # Neighbor consistency
            cons_imp = importance_results.get('neighbor_consistency_importance', {})
            if 'feature_ranking' in cons_imp:
                rankings['consistency'] = cons_imp['feature_ranking']
            
            if not rankings:
                return {'error': 'No valid importance rankings available'}
            
            # Get all unique features
            all_features = set()
            for ranking in rankings.values():
                all_features.update(ranking)
            all_features = list(all_features)
            
            # Calculate average rank for each feature
            feature_scores = {}
            for feature in all_features:
                ranks = []
                for method, ranking in rankings.items():
                    if feature in ranking:
                        rank = ranking.index(feature)
                        # Convert to score (higher is better)
                        score = len(ranking) - rank
                        ranks.append(score)
                
                if ranks:
                    feature_scores[feature] = {
                        'average_score': np.mean(ranks),
                        'score_std': np.std(ranks),
                        'methods_count': len(ranks),
                        'consensus_strength': 1 - (np.std(ranks) / np.mean(ranks)) if np.mean(ranks) > 0 else 0
                    }
            
            # Create consensus ranking
            consensus_ranking = sorted(feature_scores.keys(), 
                                     key=lambda f: feature_scores[f]['average_score'], 
                                     reverse=True)
            
            return {
                'individual_rankings': rankings,
                'feature_scores': feature_scores,
                'consensus_ranking': consensus_ranking,
                'top_features': consensus_ranking[:5],
                'ranking_methods_used': list(rankings.keys()),
                'ranking_agreement': self._calculate_ranking_agreement(rankings)
            }
            
        except Exception as e:
            logger.error(f"❌ Combining importance rankings failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_ranking_agreement(self, rankings: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate agreement between different ranking methods."""
        try:
            if len(rankings) < 2:
                return {'error': 'Need at least 2 rankings for agreement calculation'}
            
            # Calculate Kendall's tau for all pairs
            from scipy.stats import kendalltau
            
            agreements = {}
            ranking_methods = list(rankings.keys())
            
            for i, method1 in enumerate(ranking_methods):
                agreements[method1] = {}
                for j, method2 in enumerate(ranking_methods):
                    if i != j:
                        # Find common features
                        common_features = set(rankings[method1]) & set(rankings[method2])
                        
                        if len(common_features) > 1:
                            # Create rank arrays for common features
                            ranks1 = [rankings[method1].index(f) for f in common_features]
                            ranks2 = [rankings[method2].index(f) for f in common_features]
                            
                            # Calculate Kendall's tau
                            tau, p_value = kendalltau(ranks1, ranks2)
                            agreements[method1][method2] = {
                                'kendall_tau': tau,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'common_features': len(common_features)
                            }
                        else:
                            agreements[method1][method2] = {'error': 'Insufficient common features'}
            
            # Calculate overall agreement
            all_taus = []
            for method1_data in agreements.values():
                for method2_data in method1_data.values():
                    if 'kendall_tau' in method2_data and not np.isnan(method2_data['kendall_tau']):
                        all_taus.append(method2_data['kendall_tau'])
            
            overall_agreement = {
                'mean_kendall_tau': np.mean(all_taus) if all_taus else 0,
                'agreement_std': np.std(all_taus) if all_taus else 0,
                'agreement_level': 'high' if np.mean(all_taus) > 0.7 else 'medium' if np.mean(all_taus) > 0.3 else 'low' if all_taus else 'unknown'
            }
            
            return {
                'pairwise_agreements': agreements,
                'overall_agreement': overall_agreement
            }
            
        except Exception as e:
            logger.error(f"❌ Ranking agreement calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_feature_recommendations(self, importance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on feature importance analysis."""
        try:
            recommendations = []
            
            # From combined ranking
            combined = importance_results.get('combined_ranking', {})
            if 'consensus_ranking' in combined:
                top_features = combined.get('top_features', [])
                if len(top_features) >= 3:
                    recommendations.append({
                        'type': 'feature_selection',
                        'message': f'Top {len(top_features)} features show consistent importance: {", ".join(top_features)}',
                        'features': top_features,
                        'confidence': 'high'
                    })
                
                # Check ranking agreement
                agreement = combined.get('ranking_agreement', {}).get('overall_agreement', {})
                if agreement.get('agreement_level') == 'low':
                    recommendations.append({
                        'type': 'ranking_disagreement',
                        'message': 'Different importance methods disagree - feature importance may be context-dependent',
                        'confidence': 'medium'
                    })
            
            # From local variation analysis
            local_var = importance_results.get('local_importance_variation', {})
            if 'most_variable_feature' in local_var:
                most_variable = local_var['most_variable_feature']
                recommendations.append({
                    'type': 'local_variation',
                    'message': f'Feature {most_variable} shows high importance variation across data regions',
                    'feature': most_variable,
                    'confidence': 'medium'
                })
            
            # From permutation importance
            perm_imp = importance_results.get('permutation_importance', {})
            if 'significant_features' in perm_imp:
                sig_features = perm_imp['significant_features']
                if len(sig_features) < 3:
                    recommendations.append({
                        'type': 'feature_reduction',
                        'message': f'Only {len(sig_features)} features show significant importance - consider feature reduction',
                        'significant_features': sig_features,
                        'confidence': 'medium'
                    })
                elif len(sig_features) > len(perm_imp.get('top_features', [])) * 0.8:
                    recommendations.append({
                        'type': 'feature_density',
                        'message': 'Most features show significant importance - model uses diverse information',
                        'confidence': 'low'
                    })
            
            return {
                'recommendations': recommendations,
                'summary': self._create_importance_summary(importance_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Feature recommendations generation failed: {str(e)}")
            return {'error': str(e)}
    
    def _create_importance_summary(self, importance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of feature importance analysis."""
        try:
            summary = {
                'analysis_methods_used': [],
                'total_features_analyzed': 0,
                'top_features_identified': [],
                'importance_consistency': 'unknown'
            }
            
            # Count analysis methods
            for method in ['permutation_importance', 'drop_column_importance', 
                          'distance_based_importance', 'neighbor_consistency_importance']:
                if method in importance_results and 'error' not in importance_results[method]:
                    summary['analysis_methods_used'].append(method)
            
            # Get total features
            combined = importance_results.get('combined_ranking', {})
            if 'consensus_ranking' in combined:
                summary['total_features_analyzed'] = len(combined['consensus_ranking'])
                summary['top_features_identified'] = combined.get('top_features', [])
                
                # Consistency from agreement analysis
                agreement = combined.get('ranking_agreement', {}).get('overall_agreement', {})
                summary['importance_consistency'] = agreement.get('agreement_level', 'unknown')
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Importance summary creation failed: {str(e)}")
            return {'error': str(e)}
    
    # ==================== COMPARISON ANALYSIS ====================
    
    def compare_with_other_methods(self) -> Dict[str, Any]:
        """
        Compare KNN with other regression methods.
        
        Returns:
        --------
        Dict[str, Any]
            Comparison analysis results
        """
        try:
            if not self.core.is_fitted:
                return {'error': 'Core model must be fitted before analysis'}
            
            cache_key = 'method_comparison'
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
            
            logger.info("🔍 Starting method comparison analysis...")
            
            X, y = self.core.X_train_scaled_, self.core.y_train_
            
            results = {
                'instance_based_comparison': self._compare_instance_based_methods(X, y),
                'global_methods_comparison': self._compare_global_methods(X, y),
                'ensemble_comparison': self._compare_ensemble_methods(X, y),
                'performance_summary': self._create_comparison_summary()
            }
            
            # Cache results
            self._analysis_cache[cache_key] = results
            
            logger.info("✅ Method comparison analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Method comparison analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _compare_instance_based_methods(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compare KNN with other instance-based methods."""
        try:
            from sklearn.neighbors import RadiusNeighborsRegressor
            
            methods = {}
            
            # Current KNN
            current_knn_scores = cross_val_score(self.core.model, X, y, cv=3, scoring='r2')
            methods['current_knn'] = {
                'model_type': 'K-Nearest Neighbors',
                'parameters': self.core.get_hyperparameter_config(),
                'cv_scores': current_knn_scores.tolist(),
                'mean_score': current_knn_scores.mean(),
                'std_score': current_knn_scores.std()
            }
            
            # Different K values
            for k in [3, 7, 10, 15]:
                if k != self.core.n_neighbors and k < len(X):
                    try:
                        knn_k = KNeighborsRegressor(
                            n_neighbors=k,
                            weights=self.core.weights,
                            algorithm=self.core.algorithm,
                            metric=self.core.metric
                        )
                        scores = cross_val_score(knn_k, X, y, cv=3, scoring='r2')
                        methods[f'knn_k{k}'] = {
                            'model_type': f'KNN (k={k})',
                            'cv_scores': scores.tolist(),
                            'mean_score': scores.mean(),
                            'std_score': scores.std()
                        }
                    except Exception as e:
                        methods[f'knn_k{k}'] = {'error': str(e)}
            
            # Different weights
            if self.core.weights == 'uniform':
                try:
                    knn_distance = KNeighborsRegressor(
                        n_neighbors=self.core.n_neighbors,
                        weights='distance',
                        algorithm=self.core.algorithm,
                        metric=self.core.metric
                    )
                    scores = cross_val_score(knn_distance, X, y, cv=3, scoring='r2')
                    methods['knn_distance_weighted'] = {
                        'model_type': 'KNN (distance weighted)',
                        'cv_scores': scores.tolist(),
                        'mean_score': scores.mean(),
                        'std_score': scores.std()
                    }
                except Exception as e:
                    methods['knn_distance_weighted'] = {'error': str(e)}
            
            # Radius Neighbors (if available)
            try:
                # Estimate radius from current KNN
                neighbors_info = self.core.get_neighbors_info(X[:min(20, len(X))])
                if 'neighbor_distances' in neighbors_info:
                    avg_distance = np.mean(neighbors_info['neighbor_distances'])
                    radius = avg_distance * 1.2  # Slightly larger than average neighbor distance
                    
                    radius_reg = RadiusNeighborsRegressor(
                        radius=radius,
                        weights=self.core.weights,
                        algorithm=self.core.algorithm,
                        metric=self.core.metric
                    )
                    scores = cross_val_score(radius_reg, X, y, cv=3, scoring='r2')
                    methods['radius_neighbors'] = {
                        'model_type': f'Radius Neighbors (r={radius:.3f})',
                        'cv_scores': scores.tolist(),
                        'mean_score': scores.mean(),
                        'std_score': scores.std()
                    }
            except Exception as e:
                methods['radius_neighbors'] = {'error': str(e)}
            
            # Find best method
            valid_methods = {k: v for k, v in methods.items() if 'error' not in v}
            if valid_methods:
                best_method = max(valid_methods.keys(), key=lambda k: valid_methods[k]['mean_score'])
                best_score = valid_methods[best_method]['mean_score']
                current_score = methods['current_knn']['mean_score']
                improvement = best_score - current_score
            else:
                best_method = 'current_knn'
                improvement = 0
            
            return {
                'methods_compared': methods,
                'best_method': best_method,
                'improvement_potential': improvement,
                'comparison_summary': {
                    'total_methods': len(methods),
                    'successful_comparisons': len(valid_methods),
                    'best_performing': best_method
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Instance-based comparison failed: {str(e)}")
            return {'error': str(e)}
    
    def _compare_global_methods(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compare KNN with global regression methods."""
        try:
            methods = {}
            
            # Current KNN score
            current_score = cross_val_score(self.core.model, X, y, cv=3, scoring='r2').mean()
            
            # Global methods to compare
            global_models = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0),
                'decision_tree': DecisionTreeRegressor(random_state=42),
                'random_forest': RandomForestRegressor(n_estimators=50, random_state=42)
            }
            
            for name, model in global_models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                    methods[name] = {
                        'model_type': name.replace('_', ' ').title(),
                        'cv_scores': scores.tolist(),
                        'mean_score': scores.mean(),
                        'std_score': scores.std(),
                        'vs_knn_improvement': scores.mean() - current_score
                    }
                except Exception as e:
                    methods[name] = {'error': str(e)}
            
            # Add current KNN for comparison
            knn_scores = cross_val_score(self.core.model, X, y, cv=3, scoring='r2')
            methods['knn_current'] = {
                'model_type': 'K-Nearest Neighbors (Current)',
                'cv_scores': knn_scores.tolist(),
                'mean_score': knn_scores.mean(),
                'std_score': knn_scores.std(),
                'vs_knn_improvement': 0.0
            }
            
            # Analysis
            valid_methods = {k: v for k, v in methods.items() if 'error' not in v}
            if valid_methods:
                best_global = max([k for k in valid_methods.keys() if k != 'knn_current'], 
                                key=lambda k: valid_methods[k]['mean_score'])
                
                knn_rank = sorted(valid_methods.keys(), 
                                key=lambda k: valid_methods[k]['mean_score'], reverse=True).index('knn_current') + 1
                
                analysis = {
                    'knn_vs_global_performance': 'better' if knn_rank == 1 else 'competitive' if knn_rank <= 2 else 'worse',
                    'knn_rank': knn_rank,
                    'total_methods': len(valid_methods),
                    'best_global_method': best_global,
                    'best_global_score': valid_methods[best_global]['mean_score'],
                    'knn_score': current_score,
                    'performance_gap': valid_methods[best_global]['mean_score'] - current_score
                }
            else:
                analysis = {'error': 'No valid global method comparisons'}
            
            return {
                'methods_compared': methods,
                'analysis': analysis,
                'recommendations': self._generate_global_comparison_recommendations(methods, analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ Global methods comparison failed: {str(e)}")
            return {'error': str(e)}
    
    def _compare_ensemble_methods(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compare with ensemble methods."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
            from sklearn.linear_model import ElasticNet
            
            methods = {}
            current_score = cross_val_score(self.core.model, X, y, cv=3, scoring='r2').mean()
            
            ensemble_models = {
                'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'extra_trees': ExtraTreesRegressor(n_estimators=50, random_state=42),
                'elastic_net': ElasticNet(alpha=0.1)
            }
            
            for name, model in ensemble_models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                    methods[name] = {
                        'model_type': name.replace('_', ' ').title(),
                        'cv_scores': scores.tolist(),
                        'mean_score': scores.mean(),
                        'std_score': scores.std(),
                        'vs_knn_improvement': scores.mean() - current_score
                    }
                except Exception as e:
                    methods[name] = {'error': str(e)}
            
            # Find best ensemble method
            valid_methods = {k: v for k, v in methods.items() if 'error' not in v}
            if valid_methods:
                best_ensemble = max(valid_methods.keys(), key=lambda k: valid_methods[k]['mean_score'])
                best_improvement = valid_methods[best_ensemble]['vs_knn_improvement']
                
                ensemble_analysis = {
                    'best_ensemble_method': best_ensemble,
                    'best_ensemble_score': valid_methods[best_ensemble]['mean_score'],
                    'improvement_over_knn': best_improvement,
                    'knn_competitive': best_improvement < 0.05  # Within 5% is considered competitive
                }
            else:
                ensemble_analysis = {'error': 'No valid ensemble comparisons'}
            
            return {
                'ensemble_methods': methods,
                'analysis': ensemble_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Ensemble methods comparison failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_global_comparison_recommendations(self, methods: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on global method comparison."""
        try:
            recommendations = []
            
            if 'error' in analysis:
                return recommendations
            
            performance = analysis.get('knn_vs_global_performance', 'unknown')
            
            if performance == 'worse':
                gap = analysis.get('performance_gap', 0)
                best_method = analysis.get('best_global_method', 'unknown')
                
                if gap > 0.1:  # Significant gap
                    recommendations.append({
                        'type': 'method_change',
                        'message': f'Consider switching to {best_method} - shows {gap:.3f} improvement',
                        'confidence': 'high',
                        'alternative_method': best_method
                    })
                else:
                    recommendations.append({
                        'type': 'parameter_tuning',
                        'message': f'KNN slightly underperforms - try parameter optimization',
                        'confidence': 'medium'
                    })
            
            elif performance == 'better':
                recommendations.append({
                    'type': 'method_validation',
                    'message': 'KNN outperforms global methods - good choice for this dataset',
                    'confidence': 'high'
                })
            
            elif performance == 'competitive':
                recommendations.append({
                    'type': 'method_consideration',
                    'message': 'KNN is competitive but consider ensemble methods for potential improvement',
                    'confidence': 'medium'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Global comparison recommendations failed: {str(e)}")
            return []
    
    def _create_comparison_summary(self) -> Dict[str, Any]:
        """Create overall comparison summary."""
        try:
            return {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'knn_configuration': self.core.get_hyperparameter_config(),
                'comparison_scope': [
                    'instance_based_methods',
                    'global_methods', 
                    'ensemble_methods'
                ],
                'evaluation_metric': 'r2_score',
                'cross_validation_folds': 3
            }
            
        except Exception as e:
            logger.error(f"❌ Comparison summary creation failed: {str(e)}")
            return {'error': str(e)}
    
    # ==================== UTILITY METHODS ====================
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._analysis_cache.clear()
        logger.info("🗑️ Analysis cache cleared")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of available analyses."""
        return {
            'cached_analyses': list(self._analysis_cache.keys()),
            'analysis_methods_available': [
                'k_optimization',
                'distance_metrics',
                'algorithm_efficiency',
                'cross_validation',
                'feature_importance',
                'method_comparison'
            ],
            'core_model_fitted': self.core.is_fitted,
            'configuration': self.config
        }
    
    def __repr__(self) -> str:
        """String representation of the KNN Analysis."""
        status = "ready" if self.core.is_fitted else "waiting for fitted model"
        cached = len(self._analysis_cache)
        return f"KNNAnalysis(status={status}, cached_analyses={cached})"