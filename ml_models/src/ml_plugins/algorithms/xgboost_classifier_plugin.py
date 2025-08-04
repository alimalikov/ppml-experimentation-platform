import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# XGBoost imports with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

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

class XGBoostClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    XGBoost Classifier Plugin - Industry Standard Gradient Boosting
    
    XGBoost (Extreme Gradient Boosting) is the gold standard for machine learning
    competitions and industry applications. It provides superior performance through
    advanced gradient boosting with extensive optimizations and regularization.
    """
    
    def __init__(self, 
                 objective='multi:softprob',
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.3,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 colsample_bylevel=1.0,
                 colsample_bynode=1.0,
                 reg_alpha=0.0,
                 reg_lambda=1.0,
                 gamma=0.0,
                 min_child_weight=1,
                 max_delta_step=0,
                 scale_pos_weight=1.0,
                 grow_policy='depthwise',
                 max_leaves=0,
                 max_bin=256,
                 num_parallel_tree=1,
                 monotone_constraints=None,
                 interaction_constraints=None,
                 importance_type='gain',
                 gpu_id=-1,
                 validate_parameters=True,
                 predictor='auto',
                 enable_categorical=False,
                 feature_types=None,
                 max_cat_to_onehot=4,
                 eval_metric=None,
                 early_stopping_rounds=None,
                 callbacks=None,
                 random_state=42,
                 n_jobs=1,
                 verbosity=1):
        """
        Initialize XGBoost Classifier with comprehensive parameter support
        
        Parameters cover all major XGBoost functionality including:
        - Core boosting parameters (n_estimators, learning_rate, etc.)
        - Tree structure control (max_depth, min_child_weight, etc.)
        - Regularization (reg_alpha, reg_lambda, gamma)
        - Sampling parameters (subsample, colsample_*)
        - Advanced features (monotone constraints, categorical support)
        - Performance optimization (gpu_id, n_jobs, predictor)
        """
        super().__init__()
        
        # Check XGBoost availability
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
        
        # Core parameters
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        
        # Regularization parameters
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.scale_pos_weight = scale_pos_weight
        
        # Tree construction parameters
        self.grow_policy = grow_policy
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.num_parallel_tree = num_parallel_tree
        
        # Advanced parameters
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters
        self.predictor = predictor
        self.enable_categorical = enable_categorical
        self.feature_types = feature_types
        self.max_cat_to_onehot = max_cat_to_onehot
        
        # Training parameters
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.callbacks = callbacks
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        
        # Plugin metadata
        self._name = "XGBoost"
        self._description = "Industry-standard extreme gradient boosting with advanced optimizations and regularization techniques."
        self._category = "Tree-Based Models"
        self._algorithm_type = "Optimized Gradient Boosting"
        self._paper_reference = "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 30
        self._handles_missing_values = True
        self._requires_scaling = False
        self._supports_sparse = True
        self._is_linear = False
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = True
        self._ensemble_method = True
        self._supports_gpu = True
        self._industry_standard = True
        self._competition_winner = True
        self._highly_optimized = True
        self._advanced_regularization = True
        self._supports_early_stopping = True
        self._handles_imbalanced_data = True
        
        # Internal attributes
        self.model_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.evals_result_ = None
        
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
            "version": xgb.__version__ if XGBOOST_AVAILABLE else "Not Available",
            "industry_status": "Gold Standard",
            "competition_dominance": {
                "kaggle_wins": "Hundreds of competition victories",
                "industry_adoption": "Widespread use in production",
                "academic_citations": "Thousands of research papers",
                "benchmark_performance": "Consistently top-tier results"
            },
            "key_innovations": {
                "optimized_objective": "Second-order gradient optimization",
                "regularization": "Built-in L1/L2 regularization",
                "missing_values": "Native missing value handling",
                "sparsity_awareness": "Optimized for sparse data",
                "parallel_training": "Multi-threaded tree construction",
                "cache_optimization": "Cache-aware access patterns",
                "memory_efficiency": "Block-wise computation",
                "categorical_support": "Native categorical feature handling"
            },
            "strengths": [
                "Industry-standard performance",
                "Extremely well-optimized implementation",
                "Native missing value handling",
                "Advanced regularization techniques",
                "GPU acceleration support",
                "Excellent memory efficiency",
                "Handles categorical features natively",
                "Built-in cross-validation and early stopping",
                "Extensive hyperparameter control",
                "Battle-tested in production environments",
                "Active development and community",
                "Comprehensive feature importance metrics",
                "Monotone constraint support",
                "Interaction constraint capabilities",
                "Multiple objective functions",
                "Distributed training support"
            ],
            "weaknesses": [
                "Many hyperparameters to tune",
                "Can be overkill for simple problems",
                "Requires XGBoost library installation",
                "Memory intensive for very large datasets",
                "Can easily overfit without proper regularization",
                "Longer training time than simpler algorithms",
                "Black-box nature (less interpretable than single trees)"
            ],
            "use_cases": [
                "Kaggle competitions and data science contests",
                "Production machine learning systems",
                "Financial risk assessment and fraud detection",
                "Healthcare diagnosis and prognosis",
                "Marketing response and customer segmentation",
                "Recommender systems and ranking",
                "Computer vision feature-based classification",
                "Natural language processing with engineered features",
                "IoT and sensor data analysis",
                "Supply chain optimization",
                "Quality control and defect detection",
                "Any high-stakes prediction problem"
            ],
            "algorithmic_details": {
                "optimization": "Second-order Taylor expansion of loss function",
                "tree_learning": "Level-wise or leaf-wise tree construction",
                "regularization_terms": "L1 (alpha) and L2 (lambda) penalties",
                "missing_value_handling": "Learned optimal default directions",
                "sparsity_handling": "Sparse-aware split finding",
                "memory_optimization": "Block-based data layout",
                "parallel_computation": "Multi-threaded tree construction",
                "cache_efficiency": "Cache-conscious access patterns"
            },
            "performance_characteristics": {
                "training_speed": "Very fast with optimizations",
                "prediction_speed": "Extremely fast",
                "memory_usage": "Moderate to high",
                "scalability": "Excellent with proper hardware",
                "gpu_acceleration": "Significant speedup available"
            },
            "comparison_with_competitors": {
                "vs_lightgbm": "Similar performance, XGB more mature",
                "vs_catboost": "XGB faster, CatBoost better with categories",
                "vs_sklearn_gb": "XGB significantly faster and more features",
                "vs_random_forest": "XGB usually higher accuracy, more tuning required"
            },
            "hyperparameter_categories": {
                "boosting_params": ["n_estimators", "learning_rate", "objective"],
                "tree_params": ["max_depth", "min_child_weight", "gamma"],
                "regularization": ["reg_alpha", "reg_lambda", "scale_pos_weight"],
                "sampling": ["subsample", "colsample_bytree", "colsample_bylevel"],
                "performance": ["n_jobs", "gpu_id", "predictor"],
                "advanced": ["monotone_constraints", "interaction_constraints"]
            }
        }
    
    def fit(self, X, y, 
            eval_set=None, 
            eval_metric=None,
            early_stopping_rounds=None,
            verbose=True,
            xgb_model=None,
            sample_weight=None,
            sample_weight_eval_set=None,
            feature_weights=None,
            callbacks=None):
    
        # IMPORTANT: Store feature names BEFORE any processing
        print(f"DEBUG FIT - Input X type: {type(X)}")
        print(f"DEBUG FIT - Input X shape: {X.shape}")
        
        if hasattr(X, 'columns') and X.columns is not None:
            self.feature_names_ = list(X.columns)
            print(f"DEBUG FIT - Stored feature names from X.columns: {self.feature_names_}")
        else:
            print(f"DEBUG FIT - X has no columns attribute, creating default names")
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Validate input - this might convert DataFrame to numpy array
        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        
        print(f"DEBUG FIT - After check_X_y, X type: {type(X)}")
        print(f"DEBUG FIT - Feature names stored: {self.feature_names_}")
        
        # Store training info
        self.n_features_in_ = X.shape[1]
        
        # Encode labels if they're not numeric
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        
        # Determine objective based on number of classes
        n_classes = len(self.classes_)
        if n_classes == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softprob'
        
        # Create XGBoost classifier
        self.model_ = xgb.XGBClassifier(
            objective=objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            scale_pos_weight=self.scale_pos_weight,
            grow_policy=self.grow_policy,
            max_leaves=self.max_leaves,
            max_bin=self.max_bin,
            num_parallel_tree=self.num_parallel_tree,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            importance_type=self.importance_type,
            gpu_id=self.gpu_id,
            validate_parameters=self.validate_parameters,
            predictor=self.predictor,
            enable_categorical=self.enable_categorical,
            feature_types=self.feature_types,
            max_cat_to_onehot=self.max_cat_to_onehot,
            eval_metric=eval_metric or self.eval_metric,
            early_stopping_rounds=early_stopping_rounds or self.early_stopping_rounds,
            callbacks=callbacks or self.callbacks,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity
        )
        
        # Train the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            fit_params = {}
            if eval_set is not None:
                # Encode eval set labels
                eval_set_encoded = []
                for eval_X, eval_y in eval_set:
                    eval_y_encoded = self.label_encoder_.transform(eval_y)
                    eval_set_encoded.append((eval_X, eval_y_encoded))
                fit_params['eval_set'] = eval_set_encoded
            
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            if sample_weight_eval_set is not None:
                fit_params['sample_weight_eval_set'] = sample_weight_eval_set
            if feature_weights is not None:
                fit_params['feature_weights'] = feature_weights
            if verbose is not None:
                fit_params['verbose'] = verbose
            if xgb_model is not None:
                fit_params['xgb_model'] = xgb_model
            
            self.model_.fit(X, y_encoded, **fit_params)
        
        # Store evaluation results
        if hasattr(self.model_, 'evals_result_'):
            self.evals_result_ = self.model_.evals_result_
        
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
        X = check_array(X, accept_sparse=True, dtype=None)
        
        # Make predictions
        y_pred_encoded = self.model_.predict(X)
        
        # Decode labels back to original format
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
        X = check_array(X, accept_sparse=True, dtype=None)
        
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self, importance_type=None) -> Optional[np.ndarray]:
        """
        Get feature importance
        
        Parameters:
        -----------
        importance_type : str, optional
            Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
            
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores
        """
        if not self.is_fitted_:
            return None
        
        importance_type = importance_type or self.importance_type
        
        try:
            return self.model_.feature_importances_
        except:
            # Fallback to booster importance
            booster = self.model_.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)
            
            # Create array with proper ordering
            importance = np.zeros(self.n_features_in_)
            for i, feature_name in enumerate(self.feature_names_):
                importance[i] = importance_dict.get(feature_name, 0.0)
            
            return importance
    
    def get_booster(self):
        """Get the underlying XGBoost Booster object"""
        if not self.is_fitted_:
            return None
        return self.model_.get_booster()
    
    def get_evaluation_results(self) -> Optional[Dict]:
        """
        Get evaluation results from training
        
        Returns:
        --------
        results : dict
            Evaluation results by metric and dataset
        """
        return self.evals_result_
    
    def get_xgboost_analysis(self) -> Dict[str, Any]:
        """
        Analyze XGBoost-specific features and performance
        
        Returns:
        --------
        analysis : dict
            XGBoost-specific analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "xgboost_version": xgb.__version__,
            "model_type": "XGBClassifier",
            "objective": self.model_.objective,
            "n_estimators": self.model_.n_estimators,
            "gpu_enabled": self.gpu_id >= 0,
            "early_stopping_used": self.early_stopping_rounds is not None
        }
        
        # Booster information
        booster = self.get_booster()
        if booster:
            analysis["booster_info"] = {
                "num_features": booster.num_features(),
                "num_boosted_rounds": booster.num_boosted_rounds(),
                "best_iteration": getattr(self.model_, 'best_iteration', None),
                "best_score": getattr(self.model_, 'best_score', None)
            }
        
        # Feature importance types
        importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        analysis["importance_types"] = {}
        
        for imp_type in importance_types:
            try:
                importance = self.get_feature_importance(imp_type)
                if importance is not None:
                    analysis["importance_types"][imp_type] = {
                        "available": True,
                        "max_importance": float(np.max(importance)),
                        "min_importance": float(np.min(importance)),
                        "mean_importance": float(np.mean(importance))
                    }
            except:
                analysis["importance_types"][imp_type] = {"available": False}
        
        # Regularization analysis
        analysis["regularization"] = {
            "l1_alpha": self.reg_alpha,
            "l2_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "regularization_strength": "High" if (self.reg_alpha > 0.1 or self.reg_lambda > 1.5) else 
                                     "Medium" if (self.reg_alpha > 0.01 or self.reg_lambda > 0.5) else "Low"
        }
        
        # Sampling analysis
        analysis["sampling"] = {
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "colsample_bynode": self.colsample_bynode,
            "stochasticity_level": "High" if self.subsample < 0.8 else 
                                 "Medium" if self.subsample < 0.95 else "Low"
        }
        
        # Performance optimizations
        analysis["optimizations"] = {
            "parallel_jobs": self.n_jobs,
            "predictor": self.predictor,
            "tree_method": getattr(self.model_, 'tree_method', 'auto'),
            "categorical_features": self.enable_categorical,
            "gpu_acceleration": self.gpu_id >= 0
        }
        
        return analysis
    
    def plot_feature_importance(self, **kwargs):
        """Plot feature importance with flexible parameters"""
        # Extract parameters if provided
        X_test = kwargs.get('X_test')
        y_test = kwargs.get('y_test') 
        feature_names = kwargs.get('feature_names', [])
        
        # DEBUG: Print what we're getting
        print(f"DEBUG PLUGIN - feature_names passed: {feature_names}")
        print(f"DEBUG PLUGIN - self.feature_names_: {getattr(self, 'feature_names_', None)}")
        
        if not self.is_fitted_:
            return None
        
        importance = self.get_feature_importance()
        if importance is None:
            return None
        
        print(f"DEBUG PLUGIN - Importance array length: {len(importance)}")
        print(f"DEBUG PLUGIN - Importance values: {importance}")
        
        import matplotlib.pyplot as plt
        
        # Use stored feature names from training as first priority
        if hasattr(self, 'feature_names_') and self.feature_names_ and len(self.feature_names_) == len(importance):
            names = self.feature_names_
            print(f"DEBUG PLUGIN - Using self.feature_names_: {names}")
        elif feature_names and len(feature_names) == len(importance):
            names = feature_names
            print(f"DEBUG PLUGIN - Using passed feature_names: {names}")
        else:
            names = [f"Feature_{i}" for i in range(len(importance))]
            print(f"DEBUG PLUGIN - Using default names: {names}")
            print(f"DEBUG PLUGIN - Reason: self.feature_names_={getattr(self, 'feature_names_', None)}, feature_names length={len(feature_names) if feature_names else 0}, importance length={len(importance)}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by importance
        indices = np.argsort(importance)[-20:]  # Top 20 features
        
        ax.barh(range(len(indices)), importance[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('XGBoost Feature Importance')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        return fig
    
    def plot_training_curves(self, figsize=(15, 10)):
        """
        Plot training curves if evaluation results are available
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Training curves plot
        """
        if not self.is_fitted_ or self.evals_result_ is None:
            return None
        
        results = self.evals_result_
        
        # Count metrics and datasets
        datasets = list(results.keys())
        if not datasets:
            return None
        
        metrics = list(results[datasets[0]].keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Max 4 metrics
            ax = axes[i]
            
            for dataset in datasets:
                if metric in results[dataset]:
                    epochs = range(len(results[dataset][metric]))
                    values = results[dataset][metric]
                    ax.plot(epochs, values, label=f'{dataset}', linewidth=2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Mark best iteration if available
            if hasattr(self.model_, 'best_iteration') and self.model_.best_iteration is not None:
                ax.axvline(x=self.model_.best_iteration, color='red', linestyle='--', 
                          alpha=0.7, label=f'Best Iteration ({self.model_.best_iteration})')
        
        # Hide unused subplots
        for i in range(n_metrics, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_xgboost_analysis(self, figsize=(15, 10)):
        """
        Create comprehensive XGBoost analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            XGBoost analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        analysis = self.get_xgboost_analysis()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Feature importance comparison (different types)
        importance_types = ['weight', 'gain', 'cover']
        importance_data = {}
        
        for imp_type in importance_types:
            importance = self.get_feature_importance(imp_type)
            if importance is not None:
                top_5_idx = np.argsort(importance)[-5:][::-1]
                importance_data[imp_type] = importance[top_5_idx]
        
        if importance_data:
            x = np.arange(5)
            width = 0.25
            colors = ['blue', 'green', 'orange']
            
            for i, (imp_type, values) in enumerate(importance_data.items()):
                ax1.bar(x + i * width, values, width, label=imp_type, color=colors[i], alpha=0.7)
            
            top_features = [self.feature_names_[i] for i in np.argsort(self.get_feature_importance())[-5:][::-1]]
            ax1.set_xlabel('Top Features')
            ax1.set_ylabel('Importance Score')
            ax1.set_title('Feature Importance by Type (Top 5)')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(top_features, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Hyperparameter visualization
        params = {
            'Learning Rate': self.learning_rate,
            'Max Depth': self.max_depth / 20,  # Normalize
            'Subsample': self.subsample,
            'ColSample Tree': self.colsample_bytree,
            'Reg Alpha': min(self.reg_alpha, 1.0),  # Cap for visualization
            'Reg Lambda': min(self.reg_lambda / 5, 1.0)  # Normalize
        }
        
        param_names = list(params.keys())
        param_values = list(params.values())
        
        ax2.barh(param_names, param_values, color='lightblue', alpha=0.7)
        ax2.set_xlabel('Normalized Value')
        ax2.set_title('Key Hyperparameters (Normalized)')
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Regularization strength visualization
        reg_components = {
            'L1 (Alpha)': min(self.reg_alpha * 10, 1.0),  # Scale for visualization
            'L2 (Lambda)': min(self.reg_lambda / 5, 1.0),
            'Gamma': min(self.gamma * 10, 1.0),
            'Min Child Weight': min(self.min_child_weight / 10, 1.0)
        }
        
        labels = list(reg_components.keys())
        values = list(reg_components.values())
        colors = ['red', 'blue', 'green', 'orange']
        
        ax3.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Regularization Components')
        
        # 4. XGBoost optimizations summary
        if 'optimizations' in analysis:
            opt_data = analysis['optimizations']
            opt_features = ['Multi-threading', 'GPU Acceleration', 'Categorical Support', 'Optimized Predictor']
            opt_status = [
                1 if opt_data.get('parallel_jobs', 1) > 1 else 0,
                1 if opt_data.get('gpu_acceleration', False) else 0,
                1 if opt_data.get('categorical_features', False) else 0,
                1 if opt_data.get('predictor', 'auto') != 'auto' else 0.5
            ]
            
            colors = ['green' if x == 1 else 'orange' if x == 0.5 else 'red' for x in opt_status]
            ax4.bar(opt_features, opt_status, color=colors, alpha=0.7)
            ax4.set_ylabel('Enabled (1) / Disabled (0)')
            ax4.set_title('XGBoost Optimizations Status')
            ax4.set_ylim(0, 1.2)
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🏆 XGBoost Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Trees", "Regularization", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Boosting Parameters**")
            
            # Learning rate
            learning_rate = st.number_input(
                "Learning Rate:",
                value=float(self.learning_rate),
                min_value=0.001,
                max_value=1.0,
                step=0.01,
                help="Step size shrinkage. Lower = better generalization but slower convergence",
                key=f"{key_prefix}_learning_rate"
            )
            
            # Number of estimators
            n_estimators = st.slider(
                "Number of Estimators:",
                min_value=10,
                max_value=2000,
                value=int(self.n_estimators),
                step=10,
                help="Number of boosting rounds. More = better fit but risk of overfitting",
                key=f"{key_prefix}_n_estimators"
            )
            
            # Subsample
            subsample = st.slider(
                "Subsample Ratio:",
                min_value=0.1,
                max_value=1.0,
                value=float(self.subsample),
                step=0.05,
                help="Fraction of samples used per iteration. <1.0 prevents overfitting",
                key=f"{key_prefix}_subsample"
            )
            
            # Column sampling
            colsample_bytree = st.slider(
                "ColSample by Tree:",
                min_value=0.1,
                max_value=1.0,
                value=float(self.colsample_bytree),
                step=0.05,
                help="Fraction of features used per tree",
                key=f"{key_prefix}_colsample_bytree"
            )
            
            # Scale pos weight for imbalanced data
            scale_pos_weight = st.number_input(
                "Scale Pos Weight:",
                value=float(self.scale_pos_weight),
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                help="Balance of positive and negative weights. Use negative_samples/positive_samples for imbalanced data",
                key=f"{key_prefix}_scale_pos_weight"
            )
        
        with tab2:
            st.markdown("**Tree Structure Parameters**")
            
            # Max depth
            max_depth = st.slider(
                "Max Tree Depth:",
                min_value=1,
                max_value=20,
                value=int(self.max_depth),
                help="Maximum depth of trees. Higher = more complex trees",
                key=f"{key_prefix}_max_depth"
            )
            
            # Min child weight
            min_child_weight = st.slider(
                "Min Child Weight:",
                min_value=1,
                max_value=100,
                value=int(self.min_child_weight),
                help="Minimum sum of instance weight needed in a child",
                key=f"{key_prefix}_min_child_weight"
            )
            
            # Gamma
            gamma = st.number_input(
                "Gamma (Min Split Loss):",
                value=float(self.gamma),
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                help="Minimum loss reduction required to make a split",
                key=f"{key_prefix}_gamma"
            )
            
            # Max leaves
            max_leaves_enabled = st.checkbox(
                "Limit Max Leaves",
                value=self.max_leaves > 0,
                help="Limit maximum number of leaves in trees",
                key=f"{key_prefix}_max_leaves_enabled"
            )
            
            if max_leaves_enabled:
                max_leaves = st.slider(
                    "Max Leaves:",
                    min_value=2,
                    max_value=1000,
                    value=int(self.max_leaves) if self.max_leaves > 0 else 100,
                    help="Maximum number of leaves in trees",
                    key=f"{key_prefix}_max_leaves"
                )
            else:
                max_leaves = 0
            
            # Grow policy
            grow_policy = st.selectbox(
                "Tree Grow Policy:",
                options=['depthwise', 'lossguide'],
                index=['depthwise', 'lossguide'].index(self.grow_policy),
                help="depthwise: Level-order, lossguide: Leaf-wise (faster)",
                key=f"{key_prefix}_grow_policy"
            )
        
        with tab3:
            st.markdown("**Regularization Parameters**")
            
            # L1 regularization
            reg_alpha = st.number_input(
                "L1 Regularization (Alpha):",
                value=float(self.reg_alpha),
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                help="L1 regularization term on weights. Higher = more regularization",
                key=f"{key_prefix}_reg_alpha"
            )
            
            # L2 regularization
            reg_lambda = st.number_input(
                "L2 Regularization (Lambda):",
                value=float(self.reg_lambda),
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                help="L2 regularization term on weights. Higher = more regularization",
                key=f"{key_prefix}_reg_lambda"
            )
            
            # Column sampling by level
            colsample_bylevel = st.slider(
                "ColSample by Level:",
                min_value=0.1,
                max_value=1.0,
                value=float(self.colsample_bylevel),
                step=0.05,
                help="Fraction of features used per tree level",
                key=f"{key_prefix}_colsample_bylevel"
            )
            
            # Column sampling by node
            colsample_bynode = st.slider(
                "ColSample by Node:",
                min_value=0.1,
                max_value=1.0,
                value=float(self.colsample_bynode),
                step=0.05,
                help="Fraction of features used per tree node",
                key=f"{key_prefix}_colsample_bynode"
            )
        
        with tab4:
            st.markdown("**Advanced Parameters**")
            
            # Early stopping
            early_stopping_enabled = st.checkbox(
                "Enable Early Stopping",
                value=self.early_stopping_rounds is not None,
                help="Stop when validation score stops improving",
                key=f"{key_prefix}_early_stopping_enabled"
            )
            
            if early_stopping_enabled:
                early_stopping_rounds = st.slider(
                    "Early Stopping Rounds:",
                    min_value=5,
                    max_value=100,
                    value=int(self.early_stopping_rounds) if self.early_stopping_rounds else 10,
                    help="Number of rounds with no improvement before stopping",
                    key=f"{key_prefix}_early_stopping_rounds"
                )
            else:
                early_stopping_rounds = None
            
            # Importance type
            importance_type = st.selectbox(
                "Feature Importance Type:",
                options=['gain', 'weight', 'cover', 'total_gain', 'total_cover'],
                index=['gain', 'weight', 'cover', 'total_gain', 'total_cover'].index(self.importance_type),
                help="gain: Average gain, weight: Number of times used, cover: Average coverage",
                key=f"{key_prefix}_importance_type"
            )
            
            # GPU acceleration
            gpu_enabled = st.checkbox(
                "GPU Acceleration",
                value=self.gpu_id >= 0,
                help="Use GPU for training (requires GPU setup)",
                key=f"{key_prefix}_gpu_enabled"
            )
            
            if gpu_enabled:
                gpu_id = st.number_input(
                    "GPU ID:",
                    value=max(0, int(self.gpu_id)),
                    min_value=0,
                    max_value=7,
                    help="GPU device ID to use",
                    key=f"{key_prefix}_gpu_id"
                )
            else:
                gpu_id = -1
            
            # Categorical features
            enable_categorical = st.checkbox(
                "Native Categorical Support",
                value=self.enable_categorical,
                help="Enable XGBoost's native categorical feature handling",
                key=f"{key_prefix}_enable_categorical"
            )
            
            # Number of parallel trees
            num_parallel_tree = st.slider(
                "Parallel Trees:",
                min_value=1,
                max_value=10,
                value=int(self.num_parallel_tree),
                help="Number of parallel trees per iteration (for Random Forest-like behavior)",
                key=f"{key_prefix}_num_parallel_tree"
            )
            
            # Random state
            random_state = st.number_input(
                "Random State:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            st.success("""
            **XGBoost** - The Industry Standard:
            • 🏆 Winner of countless Kaggle competitions
            • 🚀 Extremely optimized implementation
            • 📊 Native missing value handling
            • 🎯 Advanced regularization techniques
            • ⚡ GPU acceleration support
            • 🔧 Extensive hyperparameter control
            
            **Key Advantages:**
            • Second-order gradient optimization
            • Built-in cross-validation
            • Multiple objective functions
            • Distributed training support
            """)
            
            # Installation check
            if XGBOOST_AVAILABLE:
                st.info(f"✅ XGBoost {xgb.__version__} is available")
            else:
                st.error("❌ XGBoost not installed. Run: pip install xgboost")
            
            # Performance tips
            if st.button("🎯 Tuning Strategy", key=f"{key_prefix}_tuning_guide"):
                st.markdown("""
                **XGBoost Tuning Strategy:**
                
                **Step 1: Basic Setup**
                - Start: n_estimators=100, learning_rate=0.3, max_depth=6
                - Use early stopping with validation set
                
                **Step 2: Tree Parameters**
                - Tune max_depth and min_child_weight together
                - max_depth: 3-10, min_child_weight: 1-10
                
                **Step 3: Learning Rate & Estimators**
                - Lower learning_rate (0.01-0.1)
                - Increase n_estimators accordingly
                
                **Step 4: Regularization**
                - Add reg_alpha and reg_lambda
                - Tune subsample and colsample_bytree
                
                **Step 5: Final Optimization**
                - Enable GPU if available
                - Use categorical features if applicable
                """)
            
            # Competition tips
            if st.button("🏆 Competition Tips", key=f"{key_prefix}_competition_tips"):
                st.markdown("""
                **Winning XGBoost Strategies:**
                
                **Feature Engineering:**
                • Create interaction features
                • Use target encoding for categories
                • Add polynomial features
                • Extract time-based features
                
                **Model Optimization:**
                • Use cross-validation for hyperparameter tuning
                • Ensemble multiple XGBoost models
                • Combine with other algorithms (LightGBM, Neural Networks)
                • Stack predictions for final submission
                
                **Advanced Techniques:**
                • Monotone constraints for domain knowledge
                • Custom objective functions
                • Multi-objective optimization
                • Pseudo-labeling for semi-supervised learning
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "objective": 'multi:softprob',  # Will be set automatically
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bynode": colsample_bynode,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "gamma": gamma,
            "min_child_weight": min_child_weight,
            "max_delta_step": 0,
            "scale_pos_weight": scale_pos_weight,
            "grow_policy": grow_policy,
            "max_leaves": max_leaves,
            "max_bin": 256,
            "num_parallel_tree": num_parallel_tree,
            "monotone_constraints": None,
            "interaction_constraints": None,
            "importance_type": importance_type,
            "gpu_id": gpu_id,
            "validate_parameters": True,
            "predictor": 'auto',
            "enable_categorical": enable_categorical,
            "feature_types": None,
            "max_cat_to_onehot": 4,
            "eval_metric": None,
            "early_stopping_rounds": early_stopping_rounds,
            "callbacks": None,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return XGBoostClassifierPlugin(
            objective=hyperparameters.get("objective", self.objective),
            n_estimators=hyperparameters.get("n_estimators", self.n_estimators),
            max_depth=hyperparameters.get("max_depth", self.max_depth),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            subsample=hyperparameters.get("subsample", self.subsample),
            colsample_bytree=hyperparameters.get("colsample_bytree", self.colsample_bytree),
            colsample_bylevel=hyperparameters.get("colsample_bylevel", self.colsample_bylevel),
            colsample_bynode=hyperparameters.get("colsample_bynode", self.colsample_bynode),
            reg_alpha=hyperparameters.get("reg_alpha", self.reg_alpha),
            reg_lambda=hyperparameters.get("reg_lambda", self.reg_lambda),
            gamma=hyperparameters.get("gamma", self.gamma),
            min_child_weight=hyperparameters.get("min_child_weight", self.min_child_weight),
            max_delta_step=hyperparameters.get("max_delta_step", self.max_delta_step),
            scale_pos_weight=hyperparameters.get("scale_pos_weight", self.scale_pos_weight),
            grow_policy=hyperparameters.get("grow_policy", self.grow_policy),
            max_leaves=hyperparameters.get("max_leaves", self.max_leaves),
            max_bin=hyperparameters.get("max_bin", self.max_bin),
            num_parallel_tree=hyperparameters.get("num_parallel_tree", self.num_parallel_tree),
            monotone_constraints=hyperparameters.get("monotone_constraints", self.monotone_constraints),
            interaction_constraints=hyperparameters.get("interaction_constraints", self.interaction_constraints),
            importance_type=hyperparameters.get("importance_type", self.importance_type),
            gpu_id=hyperparameters.get("gpu_id", self.gpu_id),
            validate_parameters=hyperparameters.get("validate_parameters", self.validate_parameters),
            predictor=hyperparameters.get("predictor", self.predictor),
            enable_categorical=hyperparameters.get("enable_categorical", self.enable_categorical),
            feature_types=hyperparameters.get("feature_types", self.feature_types),
            max_cat_to_onehot=hyperparameters.get("max_cat_to_onehot", self.max_cat_to_onehot),
            eval_metric=hyperparameters.get("eval_metric", self.eval_metric),
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds", self.early_stopping_rounds),
            callbacks=hyperparameters.get("callbacks", self.callbacks),
            random_state=hyperparameters.get("random_state", self.random_state),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            verbosity=hyperparameters.get("verbosity", self.verbosity)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        XGBoost requires minimal preprocessing:
        1. Handles missing values natively
        2. No scaling required
        3. Can handle categorical features (if enable_categorical=True)
        4. Robust to outliers
        """
        return X, y
    
    def is_compatible_with_data(self, df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
        """
        Check if algorithm is compatible with the data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target_column : str
            Name of target column
            
        Returns:
        --------
        is_compatible : bool
            Whether the algorithm can handle this data
        message : str
            Detailed compatibility message
        """
        try:
            # Check XGBoost availability first
            if not XGBOOST_AVAILABLE:
                return False, "XGBoost is not installed. Please install with: pip install xgboost"
            
            # Check if target column exists
            if target_column not in df.columns:
                return False, f"Target column '{target_column}' not found in dataset"
            
            # Check dataset size
            n_samples, n_features = df.shape
            if n_samples < self._min_samples_required:
                return False, f"Minimum {self._min_samples_required} samples required for reliable XGBoost training, got {n_samples}"
            
            # Check target variable type
            target_values = df[target_column].unique()
            n_classes = len(target_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). XGBoost works better with fewer classes."
            
            # Check for missing values (XGBoost handles these natively)
            missing_values = df.isnull().sum().sum()
            has_missing = missing_values > 0
            
            # Check feature types
            feature_columns = [col for col in df.columns if col != target_column]
            numeric_features = []
            categorical_features = []
            
            for col in feature_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_features.append(col)
                else:
                    categorical_features.append(col)
            
            # XGBoost specific advantages
            advantages = []
            considerations = []
            
            # Industry standard performance
            advantages.append("🏆 Industry-standard algorithm with proven track record")
            
            # Missing value handling
            if has_missing:
                advantages.append(f"Native missing value handling ({missing_values} missing values detected)")
            
            # Dataset size advantages
            if n_samples >= 1000:
                advantages.append(f"Good dataset size ({n_samples} samples) for XGBoost optimization")
            elif n_samples < 100:
                considerations.append(f"Small dataset ({n_samples} samples) - consider simpler models or more regularization")
            
            # Feature count optimization
            if n_features >= 10:
                advantages.append(f"Good feature count ({n_features}) for gradient boosting")
            elif n_features < 5:
                considerations.append(f"Few features ({n_features}) - may not fully utilize XGBoost power")
            
            # Class balance
            if target_column in df.columns:
                class_counts = df[target_column].value_counts()
                min_class_size = class_counts.min()
                max_class_size = class_counts.max()
                
                if max_class_size / min_class_size > 10:
                    advantages.append("Imbalanced classes detected - XGBoost handles this excellently with scale_pos_weight")
                else:
                    advantages.append("Well-balanced classes ideal for XGBoost")
            
            # Categorical features
            if len(categorical_features) > 0:
                if self.enable_categorical:
                    advantages.append(f"Native categorical support for {len(categorical_features)} categorical features")
                else:
                    considerations.append(f"Categorical features detected ({len(categorical_features)}) - enable native categorical support or encode first")
            
            # High-dimensional data
            if n_features > 100:
                advantages.append(f"High-dimensional data ({n_features} features) - XGBoost excels with feature selection")
            
            # Performance considerations
            if n_samples >= 10000:
                advantages.append("Large dataset - excellent for XGBoost performance")
                considerations.append("Consider GPU acceleration for faster training")
            
            # Competition-grade scenarios
            if n_features > 20 and n_samples > 500:
                advantages.append("Dataset characteristics ideal for competition-grade performance")
            
            # Overfitting prevention
            if n_features > n_samples / 10:
                considerations.append("High feature-to-sample ratio - enable regularization and early stopping")
            
            # GPU recommendation
            if n_samples > 5000 or n_features > 100:
                considerations.append("Large dataset/features - consider GPU acceleration if available")
            
            # Hyperparameter tuning
            tuning_suggestions = []
            if n_samples >= 1000:
                tuning_suggestions.append("Use cross-validation for hyperparameter tuning")
            if n_classes > 2:
                tuning_suggestions.append("Multi-class problem - may benefit from higher n_estimators")
            if has_missing:
                tuning_suggestions.append("Missing values present - XGBoost will handle optimally")
            
            if tuning_suggestions:
                advantages.append(f"Tuning tips: {', '.join(tuning_suggestions)}")
            
            # Performance optimization recommendations
            optimization_tips = []
            if n_samples > 10000:
                optimization_tips.append("enable early stopping")
            if len(categorical_features) > 0:
                optimization_tips.append("use native categorical support")
            if n_features > 50:
                optimization_tips.append("tune colsample_bytree for feature selection")
            
            if optimization_tips:
                considerations.append(f"Optimization tips: {', '.join(optimization_tips)}")
            
            # Version info
            advantages.append(f"XGBoost {xgb.__version__} ready for production use")
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if advantages:
                message_parts.append("🚀 XGBoost advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'objective': self.objective,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state,
            'gpu_id': self.gpu_id,
            'early_stopping_rounds': self.early_stopping_rounds
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        # Basic info
        info = {
            "status": "Fitted",
            "algorithm": "XGBoost (Extreme Gradient Boosting)",
            "version": xgb.__version__,
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": list(self.classes_),
            "feature_names": self.feature_names_,
            "n_estimators": self.n_estimators
        }
        
        # XGBoost analysis
        xgb_analysis = self.get_xgboost_analysis()
        info["xgboost_analysis"] = xgb_analysis
        
        # Feature importance (multiple types)
        importance_types = ['gain', 'weight', 'cover']
        for imp_type in importance_types:
            importance = self.get_feature_importance(imp_type)
            if importance is not None:
                # Get top 5 most important features
                top_features_idx = np.argsort(importance)[-5:][::-1]
                info[f"top_features_{imp_type}"] = [
                    {
                        "feature": self.feature_names_[idx],
                        "importance": float(importance[idx])
                    }
                    for idx in top_features_idx
                ]
        
        # Evaluation results
        if self.evals_result_ is not None:
            info["evaluation_results"] = self.evals_result_
        
        # Best iteration info
        if hasattr(self.model_, 'best_iteration') and self.model_.best_iteration is not None:
            info["early_stopping"] = {
                "best_iteration": self.model_.best_iteration,
                "best_score": getattr(self.model_, 'best_score', None)
            }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the XGBoost model.

        These metrics are derived from the model's training process,
        such as evaluation results, best iteration, and feature importance summaries.
        Parameters y_true, y_pred, y_proba are kept for API consistency but are not
        directly used as metrics are sourced from the fitted model's internal attributes.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing XGBoost-specific metrics.
        """
        if not self.is_fitted_ or not hasattr(self, 'model_') or self.model_ is None:
            return {"error": "Model not fitted. Cannot retrieve XGBoost specific metrics."}

        metrics = {}
        prefix = "xgb_"  # Prefix for XGBoost specific metrics

        # Number of boosted rounds
        try:
            booster = self.model_.get_booster()
            metrics[f"{prefix}num_boosted_rounds"] = booster.num_boosted_rounds()
        except Exception as e:
            metrics[f"{prefix}num_boosted_rounds_error"] = str(e)

        # Early stopping information
        if hasattr(self.model_, 'best_iteration') and self.model_.best_iteration is not None:
            metrics[f"{prefix}best_iteration"] = self.model_.best_iteration
            if hasattr(self.model_, 'best_score') and self.model_.best_score is not None:
                 metrics[f"{prefix}best_score"] = float(self.model_.best_score)
            metrics[f"{prefix}early_stopping_triggered"] = 1 # True
        else:
            metrics[f"{prefix}early_stopping_triggered"] = 0 # False


        # Evaluation results from training (if available)
        if hasattr(self, 'evals_result_') and self.evals_result_:
            for dataset_name, metric_dict in self.evals_result_.items():
                for metric_name, values in metric_dict.items():
                    if values:
                        # Final value of the metric for this dataset
                        metrics[f"{prefix}eval_{dataset_name}_{metric_name}_final"] = float(values[-1])
                        
                        # Value of the metric at the best iteration (if early stopping was used)
                        if hasattr(self.model_, 'best_iteration') and self.model_.best_iteration is not None:
                            if self.model_.best_iteration < len(values):
                                metrics[f"{prefix}eval_{dataset_name}_{metric_name}_at_best_iter"] = float(values[self.model_.best_iteration])
                            else: # best_iteration might be n_estimators if no early stopping happened or metric didn't improve
                                metrics[f"{prefix}eval_{dataset_name}_{metric_name}_at_best_iter"] = float(values[-1])


        # GPU usage
        metrics[f"{prefix}gpu_enabled_param"] = 1 if self.gpu_id >= 0 else 0
        try:
            tree_method = self.model_.get_params().get('tree_method', 'auto')
            if tree_method in ['gpu_hist', 'gpu_exact']: # 'gpu_hist' is common
                 metrics[f"{prefix}gpu_actually_used"] = 1
            else:
                 metrics[f"{prefix}gpu_actually_used"] = 0
        except Exception:
             metrics[f"{prefix}gpu_actually_used"] = -1 # Unknown


        # Feature importance summary (using the default importance type)
        try:
            feature_importances = self.get_feature_importance()
            if feature_importances is not None and len(feature_importances) > 0:
                metrics[f"{prefix}mean_feature_importance"] = float(np.mean(feature_importances))
                metrics[f"{prefix}max_feature_importance"] = float(np.max(feature_importances))
                metrics[f"{prefix}num_important_features"] = int(np.sum(feature_importances > 0))
            else:
                metrics[f"{prefix}feature_importance_unavailable"] = 1
        except Exception as e:
            metrics[f"{prefix}feature_importance_error"] = str(e)
            
        # Number of features used by the model
        if hasattr(self.model_, 'n_features_in_'):
            metrics[f"{prefix}n_features_in_model"] = self.model_.n_features_in_

        if not metrics:
            metrics['info'] = "No specific XGBoost metrics were extracted."
            
        return metrics

def get_plugin():
    """Factory function to get plugin instance"""
    return XGBoostClassifierPlugin()
