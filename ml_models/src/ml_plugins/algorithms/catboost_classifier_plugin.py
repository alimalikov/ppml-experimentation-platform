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

# Try to import CatBoost with graceful fallback
try:
    import ml_models.src.ml_plugins.algorithms.catboost_classifier_plugin as cb
    from ml_models.src.ml_plugins.algorithms.catboost_classifier_plugin import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None
    CatBoostClassifier = None
    Pool = None

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

class CatBoostClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    CatBoost Classifier Plugin - Expert Categorical Feature Handling
    
    CatBoost (Categorical Boosting) is a gradient boosting library developed by Yandex
    that excels at handling categorical features without preprocessing and provides
    state-of-the-art performance with minimal hyperparameter tuning.
    """
    
    def __init__(self,
                 iterations=1000,
                 learning_rate=None,  # Auto-tuned by default
                 depth=6,
                 l2_leaf_reg=3.0,
                 model_size_reg=0.5,
                 rsm=1.0,
                 loss_function='MultiClass',
                 border_count=254,
                 feature_border_type='GreedyLogSum',
                 per_float_feature_quantization=None,
                 input_borders=None,
                 output_borders=None,
                 fold_permutation_block=1,
                 od_pval=0.001,
                 od_wait=20,
                 od_type='IncToDec',
                 nan_mode='Min',
                 counter_calc_method='SkipTest',
                 leaf_estimation_iterations=None,
                 leaf_estimation_method='Newton',
                 thread_count=-1,
                 random_seed=42,
                 use_best_model=None,
                 best_model_min_trees=1,
                 verbose=False,
                 silent=True,
                 logging_level='Silent',
                 metric_period=50,
                 ctr_leaf_count_limit=None,
                 store_all_simple_ctr=None,
                 max_ctr_complexity=4,
                 has_time=False,
                 allow_const_label=None,
                 target_border=None,
                 classes_count=None,
                 class_weights=None,
                 auto_class_weights=None,
                 class_names=None,
                 one_hot_max_size=2,
                 random_strength=1.0,
                 name='experiment',
                 ignored_features=None,
                 train_dir=None,
                 custom_metric=None,
                 eval_metric=None,
                 bagging_temperature=1.0,
                 save_snapshot=None,
                 snapshot_file=None,
                 snapshot_interval=600,
                 fold_len_multiplier=2.0,
                 used_ram_limit=None,
                 gpu_ram_part=0.95,
                 pinned_memory_size=104857600,
                 allow_writing_files=True,
                 final_ctr_computation_mode='Default',
                 approx_on_full_history=False,
                 boosting_type='Plain',
                 simple_ctr=None,
                 combinations_ctr=None,
                 per_feature_ctr=None,
                 ctr_description=None,
                 ctr_border_count=50,
                 ctr_history_unit='Sample',
                 monotone_constraints=None,
                 feature_weights=None,
                 penalties_coefficient=1.0,
                 first_feature_use_penalties=None,
                 model_shrink_rate=0.0,
                 model_shrink_mode='Constant',
                 langevin=False,
                 diffusion_temperature=10000.0,
                 posterior_sampling=False,
                 boost_from_average=None,
                 text_features=None,
                 tokenizers=None,
                 dictionaries=None,
                 feature_calcers=None,
                 text_processing=None,
                 embedding_features=None,
                 # Advanced categorical handling
                 cat_features=None,
                 grow_policy='SymmetricTree',
                 min_data_in_leaf=1,
                 max_leaves=31,
                 score_function='Cosine',
                 bootstrap_type='MVS',
                 subsample=None,
                 sampling_frequency='PerTreeLevel',
                 sampling_unit='Object',
                 dev_score_calc_obj_block_size=5000000,
                 max_depth=None,
                 n_estimators=None,
                 num_boost_round=None,
                 num_trees=None,
                 colsample_bylevel=None,
                 random_state=None,
                 reg_lambda=None,
                 objective=None,
                 eta=None,
                 max_bin=None,
                 scale_pos_weight=None,
                 gpu_cat_features_storage='GpuRam',
                 data_partition='DocParallel',
                 metadata=None,
                 early_stopping_rounds=None,
                 cat_feature_params=None,
                 grow_policy_params=None,
                 feature_priors=None,
                 prediction_type='Probability',
                 task_type='CPU'):
        """
        Initialize CatBoost Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        iterations : int, default=1000
            Number of boosting iterations
        learning_rate : float, optional
            Learning rate (auto-tuned if None)
        depth : int, default=6
            Depth of trees
        l2_leaf_reg : float, default=3.0
            L2 regularization coefficient
        model_size_reg : float, default=0.5
            Model size regularization coefficient
        rsm : float, default=1.0
            Random subspace method (feature sampling ratio)
        loss_function : str, default='MultiClass'
            Loss function ('MultiClass', 'Logloss', 'CrossEntropy')
        border_count : int, default=254
            Number of borders for numerical feature discretization
        feature_border_type : str, default='GreedyLogSum'
            Border selection algorithm
        od_type : str, default='IncToDec'
            Overfitting detection type
        od_wait : int, default=20
            Number of iterations to wait after overfitting detection
        nan_mode : str, default='Min'
            Method for processing missing values
        counter_calc_method : str, default='SkipTest'
            Counter calculation method for categorical features
        thread_count : int, default=-1
            Number of threads (-1 for auto)
        random_seed : int, default=42
            Random seed for reproducibility
        verbose : bool, default=False
            Enable verbose output
        one_hot_max_size : int, default=2
            Maximum categorical feature cardinality for one-hot encoding
        random_strength : float, default=1.0
            Random strength for score perturbation
        cat_features : list, optional
            Indices or names of categorical features
        grow_policy : str, default='SymmetricTree'
            Tree growing policy ('SymmetricTree', 'Depthwise', 'Lossguide')
        bootstrap_type : str, default='MVS'
            Bootstrap type ('Bayesian', 'Bernoulli', 'MVS', 'Poisson', 'No')
        boosting_type : str, default='Plain'
            Boosting scheme ('Ordered', 'Plain')
        score_function : str, default='Cosine'
            Score function for categorical features ('Cosine', 'L2', 'NewtonCosine', 'NewtonL2')
        task_type : str, default='CPU'
            Processing unit type ('CPU', 'GPU')
        """
        super().__init__()
        
        # Core parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.model_size_reg = model_size_reg
        self.rsm = rsm
        self.loss_function = loss_function
        self.border_count = border_count
        self.feature_border_type = feature_border_type
        self.per_float_feature_quantization = per_float_feature_quantization
        self.input_borders = input_borders
        self.output_borders = output_borders
        self.fold_permutation_block = fold_permutation_block
        self.od_pval = od_pval
        self.od_wait = od_wait
        self.od_type = od_type
        self.nan_mode = nan_mode
        self.counter_calc_method = counter_calc_method
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.leaf_estimation_method = leaf_estimation_method
        self.thread_count = thread_count
        self.random_seed = random_seed
        self.use_best_model = use_best_model
        self.best_model_min_trees = best_model_min_trees
        self.verbose = verbose
        self.silent = silent
        self.logging_level = logging_level
        self.metric_period = metric_period
        self.ctr_leaf_count_limit = ctr_leaf_count_limit
        self.store_all_simple_ctr = store_all_simple_ctr
        self.max_ctr_complexity = max_ctr_complexity
        self.has_time = has_time
        self.allow_const_label = allow_const_label
        self.target_border = target_border
        self.classes_count = classes_count
        self.class_weights = class_weights
        self.auto_class_weights = auto_class_weights
        self.class_names = class_names
        self.one_hot_max_size = one_hot_max_size
        self.random_strength = random_strength
        self.name = name
        self.ignored_features = ignored_features
        self.train_dir = train_dir
        self.custom_metric = custom_metric
        self.eval_metric = eval_metric
        self.bagging_temperature = bagging_temperature
        self.save_snapshot = save_snapshot
        self.snapshot_file = snapshot_file
        self.snapshot_interval = snapshot_interval
        self.fold_len_multiplier = fold_len_multiplier
        self.used_ram_limit = used_ram_limit
        self.gpu_ram_part = gpu_ram_part
        self.pinned_memory_size = pinned_memory_size
        self.allow_writing_files = allow_writing_files
        self.final_ctr_computation_mode = final_ctr_computation_mode
        self.approx_on_full_history = approx_on_full_history
        self.boosting_type = boosting_type
        self.simple_ctr = simple_ctr
        self.combinations_ctr = combinations_ctr
        self.per_feature_ctr = per_feature_ctr
        self.ctr_description = ctr_description
        self.ctr_border_count = ctr_border_count
        self.ctr_history_unit = ctr_history_unit
        self.monotone_constraints = monotone_constraints
        self.feature_weights = feature_weights
        self.penalties_coefficient = penalties_coefficient
        self.first_feature_use_penalties = first_feature_use_penalties
        self.model_shrink_rate = model_shrink_rate
        self.model_shrink_mode = model_shrink_mode
        self.langevin = langevin
        self.diffusion_temperature = diffusion_temperature
        self.posterior_sampling = posterior_sampling
        self.boost_from_average = boost_from_average
        self.text_features = text_features
        self.tokenizers = tokenizers
        self.dictionaries = dictionaries
        self.feature_calcers = feature_calcers
        self.text_processing = text_processing
        self.embedding_features = embedding_features
        
        # Advanced categorical handling
        self.cat_features = cat_features
        self.grow_policy = grow_policy
        self.min_data_in_leaf = min_data_in_leaf
        self.max_leaves = max_leaves
        self.score_function = score_function
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.sampling_frequency = sampling_frequency
        self.sampling_unit = sampling_unit
        self.dev_score_calc_obj_block_size = dev_score_calc_obj_block_size
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.num_boost_round = num_boost_round
        self.num_trees = num_trees
        self.colsample_bylevel = colsample_bylevel
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eta = eta
        self.max_bin = max_bin
        self.scale_pos_weight = scale_pos_weight
        self.gpu_cat_features_storage = gpu_cat_features_storage
        self.data_partition = data_partition
        self.metadata = metadata
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_feature_params = cat_feature_params
        self.grow_policy_params = grow_policy_params
        self.feature_priors = feature_priors
        self.prediction_type = prediction_type
        self.task_type = task_type
        
        # Plugin metadata
        self._name = "CatBoost"
        self._description = "Advanced gradient boosting with superior categorical feature handling and minimal preprocessing requirements."
        self._category = "Tree-Based Models"
        self._algorithm_type = "Gradient Boosting Classifier"
        self._paper_reference = "Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. NIPS."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 50
        self._handles_missing_values = True
        self._requires_scaling = False
        self._supports_sparse = True
        self._is_linear = False
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = True  # BEST IN CLASS!
        self._ensemble_method = True
        self._supports_early_stopping = True
        self._supports_gpu = True
        self._overfitting_resistant = True
        self._auto_parameter_tuning = True
        self._no_preprocessing_required = True
        self._industry_grade = True
        self._research_grade = True
        
        # Internal attributes
        self.model_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.categorical_features_indices_ = None
        self.categorical_features_names_ = None
        self.eval_results_ = None
        self.feature_statistics_ = None
        
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
            "framework": "Yandex CatBoost",
            "key_innovations": {
                "ordered_boosting": "Reduces overfitting through ordered target statistics",
                "categorical_feature_combinations": "Automatic categorical feature combination generation",
                "oblivious_trees": "Symmetric trees with same split condition on each level",
                "target_statistics": "Advanced categorical encoding using target statistics",
                "minimal_parameter_tuning": "Works well with default parameters",
                "gpu_acceleration": "Efficient GPU training for large datasets"
            },
            "categorical_advantages": {
                "no_preprocessing": "No need for manual encoding of categorical features",
                "automatic_encoding": "Intelligent target-based encoding for categorical features",
                "high_cardinality": "Handles high-cardinality categorical features efficiently",
                "missing_categories": "Robust handling of unseen categories in test data",
                "feature_combinations": "Automatic generation of categorical feature combinations",
                "overfitting_prevention": "Ordered boosting prevents overfitting on categorical features",
                "memory_efficiency": "Compact representation of categorical features"
            },
            "strengths": [
                "Best-in-class categorical feature handling",
                "Excellent out-of-the-box performance",
                "Minimal hyperparameter tuning required",
                "Built-in overfitting detection and prevention",
                "Robust to missing values and outliers",
                "GPU acceleration support",
                "Automatic feature interaction detection",
                "High-quality feature importance",
                "Supports various data types natively",
                "Advanced regularization techniques",
                "Ordered boosting for better generalization",
                "Handles imbalanced datasets well"
            ],
            "weaknesses": [
                "Can be slower than LightGBM for training",
                "Large memory usage for very large datasets",
                "Less ecosystem support compared to XGBoost",
                "Newer framework with evolving API",
                "Limited customization compared to other GBMs",
                "May overfit on small datasets without proper regularization"
            ],
            "ideal_use_cases": [
                "Datasets with many categorical features",
                "High-cardinality categorical variables",
                "Mixed data types (numerical + categorical)",
                "Minimal preprocessing requirements",
                "Imbalanced classification problems",
                "Time-sensitive model development",
                "Robust baseline models",
                "Recommendation systems with categorical features",
                "Financial modeling with categorical risk factors",
                "Marketing analytics with demographic data",
                "Healthcare data with categorical diagnoses",
                "E-commerce with product categories"
            ],
            "categorical_handling_details": {
                "encoding_methods": {
                    "target_statistics": "Uses target mean/frequency for encoding",
                    "one_hot_encoding": "For low-cardinality features (controlled by one_hot_max_size)",
                    "combinations": "Automatically generates feature combinations",
                    "missing_value_handling": "Treats missing as separate category"
                },
                "overfitting_prevention": {
                    "ordered_boosting": "Different permutations for target statistics",
                    "holdout_validation": "Uses validation set for overfitting detection",
                    "regularization": "Multiple regularization techniques"
                },
                "performance_optimizations": {
                    "efficient_storage": "Compact categorical feature representation",
                    "gpu_support": "Specialized GPU kernels for categorical features",
                    "memory_optimization": "Efficient memory usage for large categorical vocabularies"
                }
            },
            "comparison_with_competitors": {
                "vs_xgboost": {
                    "categorical_handling": "Much superior - no preprocessing needed",
                    "performance": "Often better accuracy with less tuning",
                    "speed": "Slower training but faster preprocessing",
                    "ease_of_use": "Much easier for categorical data"
                },
                "vs_lightgbm": {
                    "categorical_handling": "Better handling of high-cardinality categories",
                    "overfitting": "Better overfitting prevention",
                    "speed": "Slower but more robust",
                    "accuracy": "Often higher accuracy on categorical-heavy datasets"
                },
                "vs_sklearn_gbm": {
                    "categorical_handling": "Vastly superior",
                    "performance": "Much better performance",
                    "features": "More advanced features",
                    "scalability": "Better for large datasets"
                }
            },
            "hyperparameter_guide": {
                "iterations": "500-2000, use early stopping to find optimal",
                "learning_rate": "Auto-tuned by default, 0.01-0.3 for manual tuning",
                "depth": "4-10, start with 6",
                "l2_leaf_reg": "1-10, start with 3 for regularization",
                "one_hot_max_size": "2-10, controls categorical encoding threshold",
                "bootstrap_type": "MVS (default), Bayesian for small datasets",
                "border_count": "32-254, higher for more precise splits",
                "od_wait": "10-50, patience for early stopping"
            },
            "categorical_best_practices": {
                "feature_identification": "Let CatBoost auto-detect or specify explicitly",
                "preprocessing": "Minimal - keep categorical as strings/objects",
                "missing_values": "Leave as-is, CatBoost handles them natively",
                "high_cardinality": "CatBoost excels here, no need to reduce",
                "new_categories": "Model handles unseen categories gracefully",
                "feature_engineering": "Focus on domain knowledge, let CatBoost handle encoding"
            }
        }
    
    def fit(self, X, y, 
            eval_set=None,
            sample_weight=None,
            baseline=None,
            use_best_model=None,
            verbose=None,
            logging_level=None,
            plot=False,
            column_description=None,
            verbose_eval=None,
            metric_period=None,
            silent=None,
            early_stopping_rounds=None,
            save_snapshot=None,
            snapshot_file=None,
            snapshot_interval=None,
            init_model=None):
        """
        Fit the CatBoost Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        eval_set : tuple or list of tuples, optional
            Validation dataset(s) in format (X_val, y_val)
        sample_weight : array-like, optional
            Sample weights
        baseline : array-like, optional
            Baseline predictions
        use_best_model : bool, optional
            Use best model from validation
        verbose : bool, optional
            Enable verbose output
        plot : bool, default=False
            Whether to plot training progress
        early_stopping_rounds : int, optional
            Early stopping rounds
        init_model : CatBoost model, optional
            Initial model for incremental learning
            
        Returns:
        --------
        self : object
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Please install with: pip install catboost")
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, dtype=None)
        
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
        
        # Auto-detect categorical features if not specified
        if self.cat_features is None:
            self.categorical_features_indices_ = self._detect_categorical_features(X)
        else:
            self.categorical_features_indices_ = self._process_cat_features_param(X, self.cat_features)
        
        # Store categorical feature names
        if self.categorical_features_indices_:
            if hasattr(X, 'columns'):
                self.categorical_features_names_ = [X.columns[i] for i in self.categorical_features_indices_]
            else:
                self.categorical_features_names_ = [f"cat_feature_{i}" for i in self.categorical_features_indices_]
        
        # Set up loss function based on problem type
        if n_classes == 2:
            loss_function = 'Logloss'
        else:
            loss_function = 'MultiClass'
        
        # Build parameters dictionary
        params = {
            'iterations': self.iterations,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'model_size_reg': self.model_size_reg,
            'rsm': self.rsm,
            'loss_function': loss_function,
            'border_count': self.border_count,
            'feature_border_type': self.feature_border_type,
            'od_pval': self.od_pval,
            'od_wait': self.od_wait,
            'od_type': self.od_type,
            'nan_mode': self.nan_mode,
            'counter_calc_method': self.counter_calc_method,
            'leaf_estimation_method': self.leaf_estimation_method,
            'thread_count': self.thread_count,
            'random_seed': self.random_seed,
            'verbose': verbose if verbose is not None else self.verbose,
            'logging_level': logging_level if logging_level is not None else self.logging_level,
            'metric_period': metric_period if metric_period is not None else self.metric_period,
            'max_ctr_complexity': self.max_ctr_complexity,
            'one_hot_max_size': self.one_hot_max_size,
            'random_strength': self.random_strength,
            'name': self.name,
            'bagging_temperature': self.bagging_temperature,
            'fold_len_multiplier': self.fold_len_multiplier,
            'approx_on_full_history': self.approx_on_full_history,
            'boosting_type': self.boosting_type,
            'grow_policy': self.grow_policy,
            'min_data_in_leaf': self.min_data_in_leaf,
            'max_leaves': self.max_leaves,
            'score_function': self.score_function,
            'bootstrap_type': self.bootstrap_type,
            'sampling_frequency': self.sampling_frequency,
            'sampling_unit': self.sampling_unit,
            'task_type': self.task_type,
            'allow_writing_files': False  # Prevent file creation in plugin
        }
        
        # Add learning rate if specified
        if self.learning_rate is not None:
            params['learning_rate'] = self.learning_rate
        
        # Add subsample if specified
        if self.subsample is not None:
            params['subsample'] = self.subsample
        
        # Add early stopping if specified
        if early_stopping_rounds is not None:
            params['od_wait'] = early_stopping_rounds
        elif self.early_stopping_rounds is not None:
            params['od_wait'] = self.early_stopping_rounds
        
        # Add class weights if specified
        if self.class_weights is not None:
            params['class_weights'] = self.class_weights
        elif self.auto_class_weights is not None:
            params['auto_class_weights'] = self.auto_class_weights
        
        # Add custom metrics if specified
        if self.custom_metric is not None:
            params['custom_metric'] = self.custom_metric
        if self.eval_metric is not None:
            params['eval_metric'] = self.eval_metric
        
        # Create CatBoost model
        self.model_ = CatBoostClassifier(**params)
        
        # Prepare training data
        train_pool = Pool(
            X, 
            y_encoded,
            cat_features=self.categorical_features_indices_,
            weight=sample_weight,
            baseline=baseline,
            feature_names=self.feature_names_
        )
        
        # Prepare validation data
        eval_sets = None
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            
            eval_sets = []
            for X_val, y_val in eval_set:
                y_val_encoded = self.label_encoder_.transform(y_val)
                eval_pool = Pool(
                    X_val,
                    y_val_encoded,
                    cat_features=self.categorical_features_indices_,
                    feature_names=self.feature_names_
                )
                eval_sets.append(eval_pool)
        
        # Train the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            if eval_sets:
                self.model_.fit(
                    train_pool,
                    eval_set=eval_sets,
                    use_best_model=use_best_model if use_best_model is not None else self.use_best_model,
                    verbose=verbose if verbose is not None else self.verbose,
                    plot=plot,
                    early_stopping_rounds=early_stopping_rounds if early_stopping_rounds is not None else self.early_stopping_rounds
                )
            else:
                self.model_.fit(
                    train_pool,
                    verbose=verbose if verbose is not None else self.verbose,
                    plot=plot
                )
        
        # Store evaluation results
        if hasattr(self.model_, 'get_evals_result'):
            self.eval_results_ = self.model_.get_evals_result()
        
        # Analyze categorical features
        self._analyze_categorical_features(X)
        
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
        X = check_array(X, accept_sparse=False, dtype=None)
        
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Please install with: pip install catboost")
        
        # Create prediction pool
        pred_pool = Pool(
            X,
            cat_features=self.categorical_features_indices_,
            feature_names=self.feature_names_
        )
        
        # Get predictions
        y_pred_encoded = self.model_.predict(pred_pool)
        
        # Convert back to original labels
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded.astype(int))
        
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
        X = check_array(X, accept_sparse=False, dtype=None)
        
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Please install with: pip install catboost")
        
        # Create prediction pool
        pred_pool = Pool(
            X,
            cat_features=self.categorical_features_indices_,
            feature_names=self.feature_names_
        )
        
        # Get probability predictions
        probabilities = self.model_.predict_proba(pred_pool)
        
        return probabilities
    
    def _detect_categorical_features(self, X):
        """
        Auto-detect categorical features in the dataset
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        categorical_features : list
            List of categorical feature indices
        """
        categorical_features = []
        
        if hasattr(X, 'dtypes'):
            # DataFrame with dtype information
            for i, dtype in enumerate(X.dtypes):
                if dtype == 'object' or dtype.name == 'category' or str(dtype).startswith('string'):
                    categorical_features.append(i)
                elif dtype == 'bool':
                    categorical_features.append(i)
                elif np.issubdtype(dtype, np.integer):
                    # Check if integer column has few unique values (likely categorical)
                    unique_vals = X.iloc[:, i].nunique()
                    total_vals = len(X.iloc[:, i])
                    if unique_vals <= max(10, total_vals * 0.05):  # Heuristic for categorical
                        categorical_features.append(i)
        else:
            # NumPy array - check for object dtype or low cardinality
            if X.dtype == 'object':
                categorical_features = list(range(X.shape[1]))
            else:
                for i in range(X.shape[1]):
                    unique_vals = len(np.unique(X[:, i]))
                    total_vals = X.shape[0]
                    if unique_vals <= max(10, total_vals * 0.05):
                        categorical_features.append(i)
        
        return categorical_features
    
    def _process_cat_features_param(self, X, cat_features):
        """
        Process the cat_features parameter to get indices
        
        Parameters:
        -----------
        X : array-like
            Input features
        cat_features : list
            List of categorical feature names or indices
            
        Returns:
        --------
        categorical_indices : list
            List of categorical feature indices
        """
        if cat_features is None:
            return []
        
        categorical_indices = []
        
        for feature in cat_features:
            if isinstance(feature, str):
                # Feature name
                if hasattr(X, 'columns'):
                    try:
                        idx = list(X.columns).index(feature)
                        categorical_indices.append(idx)
                    except ValueError:
                        pass  # Feature not found
                else:
                    pass  # Can't use string names with numpy arrays
            elif isinstance(feature, int):
                # Feature index
                if 0 <= feature < X.shape[1]:
                    categorical_indices.append(feature)
        
        return categorical_indices
    
    def _analyze_categorical_features(self, X):
        """
        Analyze categorical features in the dataset
        
        Parameters:
        -----------
        X : array-like
            Input features
        """
        if not self.categorical_features_indices_:
            self.feature_statistics_ = {"categorical_features": 0, "analysis": "No categorical features detected"}
            return
        
        analysis = {
            "categorical_features": len(self.categorical_features_indices_),
            "categorical_feature_names": self.categorical_features_names_,
            "categorical_feature_indices": self.categorical_features_indices_,
            "categorical_statistics": {}
        }
        
        for i, idx in enumerate(self.categorical_features_indices_):
            if hasattr(X, 'iloc'):
                feature_data = X.iloc[:, idx]
                feature_name = self.categorical_features_names_[i] if self.categorical_features_names_ else f"cat_feature_{idx}"
            else:
                feature_data = X[:, idx]
                feature_name = f"cat_feature_{idx}"
            
            unique_values = len(np.unique(feature_data))
            missing_values = np.sum(pd.isna(feature_data))
            
            analysis["categorical_statistics"][feature_name] = {
                "unique_values": unique_values,
                "missing_values": missing_values,
                "cardinality": "high" if unique_values > 100 else "medium" if unique_values > 10 else "low"
            }
        
        self.feature_statistics_ = analysis
    
    def get_feature_importance(self, importance_type='PredictionValuesChange'):
        """
        Get feature importance
        
        Parameters:
        -----------
        importance_type : str, default='PredictionValuesChange'
            Type of importance: 'PredictionValuesChange', 'LossFunctionChange', 'FeatureImportance'
            
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores
        """
        if not self.is_fitted_:
            return None
        
        try:
            if importance_type == 'FeatureImportance':
                return self.model_.feature_importances_
            else:
                return self.model_.get_feature_importance(type=importance_type)
        except Exception:
            # Fallback to default feature importance
            return self.model_.feature_importances_
    
    def get_categorical_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of categorical feature handling
        
        Returns:
        --------
        analysis_info : dict
            Information about categorical features and their handling
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "catboost_version": cb.__version__ if CATBOOST_AVAILABLE else "Not available",
            "categorical_features_detected": len(self.categorical_features_indices_) if self.categorical_features_indices_ else 0,
            "categorical_handling_strategy": {
                "encoding_method": "Target statistics with ordered boosting",
                "one_hot_threshold": self.one_hot_max_size,
                "missing_value_handling": self.nan_mode,
                "overfitting_prevention": "Ordered boosting",
                "feature_combinations": "Automatic generation up to max_ctr_complexity"
            },
            "feature_statistics": self.feature_statistics_,
            "model_parameters": {
                "counter_calc_method": self.counter_calc_method,
                "max_ctr_complexity": self.max_ctr_complexity,
                "score_function": self.score_function,
                "boosting_type": self.boosting_type,
                "border_count": self.border_count
            },
            "categorical_advantages_realized": {
                "no_preprocessing": True,
                "handles_high_cardinality": True,
                "missing_value_robustness": True,
                "overfitting_prevention": True,
                "automatic_combinations": True
            }
        }
        
        # Add detailed categorical feature analysis
        if self.categorical_features_indices_:
            analysis["categorical_features_detail"] = {
                "indices": self.categorical_features_indices_,
                "names": self.categorical_features_names_,
                "encoding_applied": "Target statistics with ordered boosting",
                "combinations_generated": f"Up to {self.max_ctr_complexity} feature combinations"
            }
        
        return analysis
    
    def plot_feature_importance(self, max_features=20, importance_type='PredictionValuesChange', figsize=(10, 8)):
        """
        Create a feature importance plot
        
        Parameters:
        -----------
        max_features : int, default=20
            Maximum number of features to display
        importance_type : str, default='PredictionValuesChange'
            Type of importance
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Feature importance plot
        """
        if not self.is_fitted_:
            return None
        
        importance = self.get_feature_importance(importance_type)
        if importance is None:
            return None
        
        # Get top features
        indices = np.argsort(importance)[::-1][:max_features]
        top_features = [self.feature_names_[i] for i in indices]
        top_importance = importance[indices]
        
        # Mark categorical features
        feature_colors = []
        for i in indices:
            if self.categorical_features_indices_ and i in self.categorical_features_indices_:
                feature_colors.append('orange')  # Categorical features in orange
            else:
                feature_colors.append('lightblue')  # Numerical features in blue
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(top_features)), top_importance, 
                      color=feature_colors, alpha=0.8, edgecolor='darkblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel(f'Feature Importance ({importance_type})')
        ax.set_title(f'Top {len(top_features)} Feature Importances - CatBoost')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='orange', label='Categorical Features'),
            Patch(facecolor='lightblue', label='Numerical Features')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(top_importance) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_categorical_analysis(self, figsize=(12, 8)):
        """
        Create categorical feature analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Categorical analysis visualization
        """
        if not self.is_fitted_ or not self.feature_statistics_:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Feature type distribution
        n_categorical = len(self.categorical_features_indices_) if self.categorical_features_indices_ else 0
        n_numerical = self.n_features_in_ - n_categorical
        
        ax1.pie([n_categorical, n_numerical], 
               labels=['Categorical', 'Numerical'], 
               colors=['orange', 'lightblue'],
               autopct='%1.1f%%',
               startangle=90)
        ax1.set_title('Feature Type Distribution')
        
        # Categorical feature cardinalities
        if self.feature_statistics_.get("categorical_statistics"):
            cat_stats = self.feature_statistics_["categorical_statistics"]
            feature_names = list(cat_stats.keys())[:10]  # Show top 10
            cardinalities = [cat_stats[name]["unique_values"] for name in feature_names]
            
            colors = ['red' if card > 100 else 'orange' if card > 10 else 'green' 
                     for card in cardinalities]
            
            bars = ax2.bar(range(len(feature_names)), cardinalities, color=colors, alpha=0.7)
            ax2.set_xlabel('Categorical Features')
            ax2.set_ylabel('Number of Unique Values')
            ax2.set_title('Categorical Feature Cardinality')
            ax2.set_xticks(range(len(feature_names)))
            ax2.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                               for name in feature_names], rotation=45, ha='right')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(cardinalities) * 0.01,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No categorical features\ndetected', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Categorical Feature Cardinality')
        
        # CatBoost advantages
        advantages = [
            'No Preprocessing', 'High Cardinality', 'Missing Values', 
            'Overfitting Prevention', 'Auto Combinations', 'Target Encoding'
        ]
        scores = [10, 10, 9, 9, 8, 10]  # CatBoost scores
        
        bars = ax3.barh(advantages, scores, color='green', alpha=0.7)
        ax3.set_xlabel('Advantage Score (1-10)')
        ax3.set_title('CatBoost Categorical Advantages')
        ax3.set_xlim(0, 10)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width}/10', ha='left', va='center', fontsize=9)
        
        # Encoding strategy comparison
        strategies = ['One-Hot', 'Label\nEncoding', 'Target\nEncoding', 'CatBoost\nEncoding']
        effectiveness = [3, 5, 7, 10]  # Relative effectiveness scores
        colors = ['lightcoral', 'orange', 'lightgreen', 'darkgreen']
        
        bars = ax4.bar(strategies, effectiveness, color=colors, alpha=0.8)
        ax4.set_ylabel('Effectiveness Score')
        ax4.set_title('Categorical Encoding Strategies')
        ax4.set_ylim(0, 10)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}/10', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🐱 CatBoost Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Categorical", "Regularization", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Iterations
            iterations = st.slider(
                "Iterations:",
                min_value=100,
                max_value=5000,
                value=int(self.iterations),
                step=50,
                help="Number of boosting iterations",
                key=f"{key_prefix}_iterations"
            )
            
            # Learning rate
            auto_lr = st.checkbox(
                "Auto Learning Rate",
                value=self.learning_rate is None,
                help="Let CatBoost automatically tune learning rate",
                key=f"{key_prefix}_auto_lr"
            )
            
            if not auto_lr:
                learning_rate = st.number_input(
                    "Learning Rate:",
                    value=0.1 if self.learning_rate is None else float(self.learning_rate),
                    min_value=0.001,
                    max_value=1.0,
                    step=0.01,
                    help="Shrinkage rate for preventing overfitting",
                    key=f"{key_prefix}_learning_rate"
                )
            else:
                learning_rate = None
            
            # Tree depth
            depth = st.slider(
                "Tree Depth:",
                min_value=1,
                max_value=16,
                value=int(self.depth),
                help="Depth of trees (4-10 recommended)",
                key=f"{key_prefix}_depth"
            )
            
            # Task type
            task_type = st.selectbox(
                "Processing Unit:",
                options=['CPU', 'GPU'],
                index=['CPU', 'GPU'].index(self.task_type),
                help="CPU for compatibility, GPU for speed (if available)",
                key=f"{key_prefix}_task_type"
            )
            
            # Bootstrap type
            bootstrap_type = st.selectbox(
                "Bootstrap Type:",
                options=['MVS', 'Bayesian', 'Bernoulli', 'Poisson', 'No'],
                index=['MVS', 'Bayesian', 'Bernoulli', 'Poisson', 'No'].index(self.bootstrap_type),
                help="MVS: default, Bayesian: for small data, Bernoulli: for large data",
                key=f"{key_prefix}_bootstrap_type"
            )
        
        with tab2:
            st.markdown("**Categorical Features**")
            
            # Auto-detect categorical features
            auto_detect_cat = st.checkbox(
                "Auto-detect Categorical Features",
                value=True,
                help="Automatically detect categorical features based on data types",
                key=f"{key_prefix}_auto_detect_cat"
            )
            
            # One-hot encoding threshold
            one_hot_max_size = st.slider(
                "One-Hot Max Size:",
                min_value=1,
                max_value=20,
                value=int(self.one_hot_max_size),
                help="Maximum cardinality for one-hot encoding (vs target encoding)",
                key=f"{key_prefix}_one_hot_max_size"
            )
            
            # Score function for categorical features
            score_function = st.selectbox(
                "Score Function:",
                options=['Cosine', 'L2', 'NewtonCosine', 'NewtonL2'],
                index=['Cosine', 'L2', 'NewtonCosine', 'NewtonL2'].index(self.score_function),
                help="Scoring function for categorical features",
                key=f"{key_prefix}_score_function"
            )
            
            # CTR complexity
            max_ctr_complexity = st.slider(
                "Max CTR Complexity:",
                min_value=1,
                max_value=8,
                value=int(self.max_ctr_complexity),
                help="Maximum complexity for categorical feature combinations",
                key=f"{key_prefix}_max_ctr_complexity"
            )
            
            # Counter calculation method
            counter_calc_method = st.selectbox(
                "Counter Calculation:",
                options=['SkipTest', 'Full'],
                index=['SkipTest', 'Full'].index(self.counter_calc_method),
                help="Method for calculating categorical feature statistics",
                key=f"{key_prefix}_counter_calc_method"
            )
            
            # Missing value handling
            nan_mode = st.selectbox(
                "Missing Value Mode:",
                options=['Min', 'Max', 'Forbidden'],
                index=['Min', 'Max', 'Forbidden'].index(self.nan_mode),
                help="How to handle missing values in categorical features",
                key=f"{key_prefix}_nan_mode"
            )
        
        with tab3:
            st.markdown("**Regularization & Overfitting**")
            
            # L2 regularization
            l2_leaf_reg = st.number_input(
                "L2 Leaf Regularization:",
                value=float(self.l2_leaf_reg),
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                help="L2 regularization coefficient (3.0 is good default)",
                key=f"{key_prefix}_l2_leaf_reg"
            )
            
            # Model size regularization
            model_size_reg = st.number_input(
                "Model Size Regularization:",
                value=float(self.model_size_reg),
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                help="Model size regularization coefficient",
                key=f"{key_prefix}_model_size_reg"
            )
            
            # Random strength
            random_strength = st.number_input(
                "Random Strength:",
                value=float(self.random_strength),
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                help="Random strength for score perturbation",
                key=f"{key_prefix}_random_strength"
            )
            
            # Overfitting detection
            od_type = st.selectbox(
                "Overfitting Detection:",
                options=['IncToDec', 'Iter'],
                index=['IncToDec', 'Iter'].index(self.od_type),
                help="Type of overfitting detection",
                key=f"{key_prefix}_od_type"
            )
            
            # Early stopping patience
            od_wait = st.slider(
                "Early Stopping Patience:",
                min_value=5,
                max_value=100,
                value=int(self.od_wait),
                help="Number of iterations to wait before stopping",
                key=f"{key_prefix}_od_wait"
            )
            
            # P-value for overfitting detection
            od_pval = st.number_input(
                "Overfitting P-value:",
                value=float(self.od_pval),
                min_value=0.0001,
                max_value=0.1,
                step=0.0001,
                format="%.4f",
                help="P-value threshold for overfitting detection",
                key=f"{key_prefix}_od_pval"
            )
        
        with tab4:
            st.markdown("**Advanced Parameters**")
            
            # Boosting type
            boosting_type = st.selectbox(
                "Boosting Type:",
                options=['Ordered', 'Plain'],
                index=['Ordered', 'Plain'].index(self.boosting_type),
                help="Ordered: better quality, Plain: faster training",
                key=f"{key_prefix}_boosting_type"
            )
            
            # Tree growing policy
            grow_policy = st.selectbox(
                "Grow Policy:",
                options=['SymmetricTree', 'Depthwise', 'Lossguide'],
                index=['SymmetricTree', 'Depthwise', 'Lossguide'].index(self.grow_policy),
                help="Tree construction strategy",
                key=f"{key_prefix}_grow_policy"
            )
            
            # RSM (Random Subspace Method)
            rsm = st.slider(
                "Random Subspace (RSM):",
                min_value=0.1,
                max_value=1.0,
                value=float(self.rsm),
                step=0.05,
                help="Random subspace method - fraction of features to use",
                key=f"{key_prefix}_rsm"
            )
            
            # Border count
            border_count = st.slider(
                "Border Count:",
                min_value=32,
                max_value=255,
                value=int(self.border_count),
                help="Number of splits for numerical features",
                key=f"{key_prefix}_border_count"
            )
            
            # Feature border type
            feature_border_type = st.selectbox(
                "Feature Border Type:",
                options=['GreedyLogSum', 'UniformAndQuantiles', 'MinEntropy', 'MaxLogSum'],
                index=['GreedyLogSum', 'UniformAndQuantiles', 'MinEntropy', 'MaxLogSum'].index(self.feature_border_type),
                help="Algorithm for selecting borders",
                key=f"{key_prefix}_feature_border_type"
            )
            
            # Bagging temperature
            bagging_temperature = st.slider(
                "Bagging Temperature:",
                min_value=0.0,
                max_value=10.0,
                value=float(self.bagging_temperature),
                step=0.1,
                help="Controls randomness in Bayesian bootstrap",
                key=f"{key_prefix}_bagging_temperature"
            )
            
            # Random seed
            random_seed = st.number_input(
                "Random Seed:",
                value=int(self.random_seed),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_seed"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if CATBOOST_AVAILABLE:
                st.success(f"✅ CatBoost {cb.__version__} is available")
            else:
                st.error("❌ CatBoost not installed. Run: pip install catboost")
            
            st.info("""
            **CatBoost** - Categorical Boosting Expert:
            • 🐱 Best categorical feature handling
            • 🎯 No preprocessing required
            • 🛡️ Built-in overfitting protection
            • ⚡ GPU acceleration support
            • 🔧 Minimal hyperparameter tuning
            • 📊 Advanced target encoding
            
            **Categorical Advantages:**
            • Handles high-cardinality categories
            • Automatic feature combinations
            • Robust to missing categories
            • Ordered boosting prevents overfitting
            """)
            
            # Categorical features guide
            if st.button("🐱 Categorical Features Guide", key=f"{key_prefix}_cat_guide"):
                st.markdown("""
                **CatBoost Categorical Feature Handling:**
                
                **Automatic Detection:**
                - String/object columns → Categorical
                - Boolean columns → Categorical  
                - Low-cardinality integers → Categorical
                
                **Encoding Strategy:**
                - Low cardinality (≤ one_hot_max_size) → One-hot
                - High cardinality → Target statistics encoding
                - Missing values → Separate category
                
                **Advanced Features:**
                - Automatic feature combinations
                - Ordered boosting prevents overfitting
                - Handles unseen categories gracefully
                """)
            
            # Tuning strategy
            if st.button("🎯 Tuning Strategy", key=f"{key_prefix}_tuning_strategy"):
                st.markdown("""
                **CatBoost Tuning Strategy:**
                
                **Step 1: Start Simple**
                - Use default parameters
                - Let auto learning rate work
                - Enable auto categorical detection
                
                **Step 2: Adjust Core Parameters**
                - Increase iterations (500-2000)
                - Tune depth (4-10)
                - Adjust l2_leaf_reg (1-10)
                
                **Step 3: Categorical Optimization**
                - Tune one_hot_max_size for your data
                - Experiment with score_function
                - Adjust max_ctr_complexity
                
                **Step 4: Overfitting Control**
                - Use validation set with early stopping
                - Increase regularization if needed
                - Consider Ordered boosting
                """)
            
            # Comparison with other GBMs
            if st.button("⚖️ vs Other GBMs", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **CatBoost vs Competitors:**
                
                **vs XGBoost:**
                - ✅ Much better categorical handling
                - ✅ No preprocessing needed
                - ✅ Better overfitting resistance
                - ❌ Slightly slower training
                
                **vs LightGBM:**
                - ✅ Superior categorical encoding
                - ✅ Better default parameters
                - ✅ More robust to overfitting
                - ❌ Higher memory usage
                
                **When to choose CatBoost:**
                - Many categorical features
                - High-cardinality categories
                - Need robust baseline quickly
                - Want minimal preprocessing
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "model_size_reg": model_size_reg,
            "rsm": rsm,
            "loss_function": 'MultiClass',  # Will be set automatically
            "border_count": border_count,
            "feature_border_type": feature_border_type,
            "od_pval": od_pval,
            "od_wait": od_wait,
            "od_type": od_type,
            "nan_mode": nan_mode,
            "counter_calc_method": counter_calc_method,
            "thread_count": -1,
            "random_seed": random_seed,
            "verbose": False,
            "silent": True,
            "logging_level": 'Silent',
            "max_ctr_complexity": max_ctr_complexity,
            "one_hot_max_size": one_hot_max_size,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "boosting_type": boosting_type,
            "grow_policy": grow_policy,
            "score_function": score_function,
            "bootstrap_type": bootstrap_type,
            "task_type": task_type,
            "cat_features": None if auto_detect_cat else [],  # Will be auto-detected
            "early_stopping_rounds": od_wait,
            "allow_writing_files": False
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return CatBoostClassifierPlugin(
            iterations=hyperparameters.get("iterations", self.iterations),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            depth=hyperparameters.get("depth", self.depth),
            l2_leaf_reg=hyperparameters.get("l2_leaf_reg", self.l2_leaf_reg),
            model_size_reg=hyperparameters.get("model_size_reg", self.model_size_reg),
            rsm=hyperparameters.get("rsm", self.rsm),
            loss_function=hyperparameters.get("loss_function", self.loss_function),
            border_count=hyperparameters.get("border_count", self.border_count),
            feature_border_type=hyperparameters.get("feature_border_type", self.feature_border_type),
            od_pval=hyperparameters.get("od_pval", self.od_pval),
            od_wait=hyperparameters.get("od_wait", self.od_wait),
            od_type=hyperparameters.get("od_type", self.od_type),
            nan_mode=hyperparameters.get("nan_mode", self.nan_mode),
            counter_calc_method=hyperparameters.get("counter_calc_method", self.counter_calc_method),
            thread_count=hyperparameters.get("thread_count", self.thread_count),
            random_seed=hyperparameters.get("random_seed", self.random_seed),
            verbose=hyperparameters.get("verbose", self.verbose),
            silent=hyperparameters.get("silent", self.silent),
            logging_level=hyperparameters.get("logging_level", self.logging_level),
            max_ctr_complexity=hyperparameters.get("max_ctr_complexity", self.max_ctr_complexity),
            one_hot_max_size=hyperparameters.get("one_hot_max_size", self.one_hot_max_size),
            random_strength=hyperparameters.get("random_strength", self.random_strength),
            bagging_temperature=hyperparameters.get("bagging_temperature", self.bagging_temperature),
            boosting_type=hyperparameters.get("boosting_type", self.boosting_type),
            grow_policy=hyperparameters.get("grow_policy", self.grow_policy),
            score_function=hyperparameters.get("score_function", self.score_function),
            bootstrap_type=hyperparameters.get("bootstrap_type", self.bootstrap_type),
            task_type=hyperparameters.get("task_type", self.task_type),
            cat_features=hyperparameters.get("cat_features", self.cat_features),
            early_stopping_rounds=hyperparameters.get("early_stopping_rounds", self.early_stopping_rounds)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for CatBoost
        
        CatBoost handles categorical features and missing values natively,
        so minimal preprocessing is needed.
        """
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
        """
        Check if CatBoost is compatible with the given data
        
        Returns:
        --------
        compatible : bool
            Whether the algorithm is compatible
        message : str
            Explanation message
        """
        if not CATBOOST_AVAILABLE:
            return False, "CatBoost is not installed. Install with: pip install catboost"
        
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"CatBoost requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            if len(unique_values) > 1000:
                return False, "Too many classes (>1000). Consider regression or multiclass strategies."
        
        return True, "CatBoost is highly compatible with this data, especially for categorical features"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "categorical_features": len(self.categorical_features_indices_) if self.categorical_features_indices_ else 0,
            "categorical_feature_names": self.categorical_features_names_,
            "boosting_type": self.boosting_type,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "tree_count": self.model_.tree_count_ if hasattr(self.model_, 'tree_count_') else None,
            "best_score": self.model_.best_score_ if hasattr(self.model_, 'best_score_') else None,
            "best_iteration": self.model_.best_iteration_ if hasattr(self.model_, 'best_iteration_') else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "CatBoost",
            "training_completed": True,
            "categorical_advantages": {
                "no_preprocessing_required": True,
                "handles_high_cardinality": True,
                "automatic_feature_combinations": True,
                "overfitting_prevention": "Ordered boosting",
                "missing_value_handling": "Native support"
            },
            "model_characteristics": {
                "boosting_type": self.boosting_type,
                "tree_structure": "Oblivious trees",
                "categorical_encoding": "Target statistics with ordered boosting",
                "regularization": "L2 + Model size + Random strength",
                "overfitting_detection": self.od_type
            },
            "categorical_features_analysis": self.get_categorical_analysis(),
            "performance_optimizations": {
                "gpu_support": self.task_type == 'GPU',
                "feature_bundling": True,
                "histogram_optimization": True,
                "memory_efficient_categorical": True
            }
        }
        
        # Add evaluation results if available
        if self.eval_results_:
            info["evaluation_results"] = self.eval_results_
        
        return info

    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate CatBoost Classifier-specific metrics based on the fitted model.

        This includes information like the number of trees built, best iteration,
        scores from the training process, and details about categorical feature handling.

        Args:
            y_true: Ground truth target values from a test set (not directly used for these model-internal metrics).
            y_pred: Predicted target values on a test set (not directly used for these model-internal metrics).
            y_proba: Predicted probabilities on a test set (not directly used for these model-internal metrics).

        Returns:
            A dictionary of CatBoost-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or not CATBOOST_AVAILABLE or self.model_ is None:
            metrics["status"] = "Model not fitted or CatBoost not available"
            return metrics

        metrics['num_trees_built'] = self.model_.tree_count_
        
        best_iter = self.model_.get_best_iteration()
        if best_iter is not None:
            metrics['best_iteration'] = int(best_iter)

        best_score_data = self.model_.get_best_score()
        if best_score_data:
            # best_score_data is like: {'learn': {'Logloss': 0.1, 'AUC': 0.9}, 'validation': {'Logloss': 0.2, 'AUC':0.88}}
            if 'learn' in best_score_data and best_score_data['learn']:
                # Take the first metric reported for learn set as primary
                primary_learn_metric_name = next(iter(best_score_data['learn']))
                metrics[f'best_score_learn_{primary_learn_metric_name}'] = best_score_data['learn'][primary_learn_metric_name]
            
            if 'validation' in best_score_data and best_score_data['validation']: # CatBoost often names the first eval set 'validation'
                primary_val_metric_name = next(iter(best_score_data['validation']))
                metrics[f'best_score_validation_{primary_val_metric_name}'] = best_score_data['validation'][primary_val_metric_name]
            elif 'validation_0' in best_score_data and best_score_data['validation_0']: # Or 'validation_0'
                primary_val_metric_name = next(iter(best_score_data['validation_0']))
                metrics[f'best_score_validation0_{primary_val_metric_name}'] = best_score_data['validation_0'][primary_val_metric_name]


        if self.categorical_features_indices_ is not None:
            metrics['num_categorical_features_handled'] = len(self.categorical_features_indices_)
        else:
            metrics['num_categorical_features_handled'] = 0
            
        metrics['loss_function_used'] = self.model_.get_params().get('loss_function', self.loss_function)
        
        eval_metric_used = self.model_.get_params().get('eval_metric')
        if eval_metric_used:
             metrics['primary_eval_metric_configured'] = eval_metric_used

        # Extract metric values from eval_results_ at best_iteration if possible
        if self.eval_results_ and best_iter is not None:
            for eval_set_name, metric_dict in self.eval_results_.items():
                for metric_name, values in metric_dict.items():
                    if best_iter < len(values):
                        metrics[f'{eval_set_name}_{metric_name}_at_best_iter'] = values[best_iter]
        
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return CatBoostClassifierPlugin()