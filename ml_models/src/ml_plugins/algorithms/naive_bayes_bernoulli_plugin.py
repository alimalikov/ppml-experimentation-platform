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

# Try to import Multinomial Naive Bayes with graceful fallback
try:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    MultinomialNB = None

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

class MultinomialNaiveBayesPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Multinomial Naive Bayes Classifier Plugin - Text & Count Features Specialist
    
    Multinomial Naive Bayes is specifically designed for features representing discrete counts,
    making it ideal for text classification, document analysis, and any application involving
    count-based features like word frequencies, n-grams, or histogram data.
    """
    
    def __init__(self,
                 alpha=1.0,
                 fit_prior=True,
                 class_prior=None,
                 # Text processing parameters
                 vectorization_method='count',
                 max_features=10000,
                 ngram_range=(1, 1),
                 min_df=1,
                 max_df=1.0,
                 stop_words=None,
                 lowercase=True,
                 # Advanced parameters
                 feature_log_prob_thresholding=False,
                 log_prob_threshold=-10.0,
                 smoothing_method='laplace',
                 feature_selection=False,
                 n_features_to_select=1000,
                 # Validation parameters
                 validate_input=True,
                 zero_feature_handling='warn'):
        """
        Initialize Multinomial Naive Bayes Classifier
        
        Parameters:
        -----------
        alpha : float, default=1.0
            Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)
        fit_prior : bool, default=True
            Whether to learn class prior probabilities
        class_prior : array-like, optional
            Prior probabilities of classes
        vectorization_method : str, default='count'
            Text vectorization method ('count', 'tfidf', 'binary')
        max_features : int, default=10000
            Maximum number of features for text vectorization
        ngram_range : tuple, default=(1, 1)
            Range of n-grams to extract (min_n, max_n)
        min_df : int/float, default=1
            Minimum document frequency for features
        max_df : float, default=1.0
            Maximum document frequency for features
        stop_words : str/list, optional
            Stop words to remove ('english', list, or None)
        lowercase : bool, default=True
            Convert text to lowercase
        feature_log_prob_thresholding : bool, default=False
            Apply thresholding to feature log probabilities
        log_prob_threshold : float, default=-10.0
            Threshold for feature log probabilities
        smoothing_method : str, default='laplace'
            Smoothing method ('laplace', 'lidstone', 'good_turing')
        feature_selection : bool, default=False
            Enable feature selection based on chi-square or mutual information
        n_features_to_select : int, default=1000
            Number of features to select
        validate_input : bool, default=True
            Validate that features are non-negative
        zero_feature_handling : str, default='warn'
            How to handle zero features ('warn', 'error', 'ignore')
        """
        super().__init__()
        
        # Core Multinomial NB parameters
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        
        # Text processing parameters
        self.vectorization_method = vectorization_method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.lowercase = lowercase
        
        # Advanced parameters
        self.feature_log_prob_thresholding = feature_log_prob_thresholding
        self.log_prob_threshold = log_prob_threshold
        self.smoothing_method = smoothing_method
        self.feature_selection = feature_selection
        self.n_features_to_select = n_features_to_select
        
        # Validation parameters
        self.validate_input = validate_input
        self.zero_feature_handling = zero_feature_handling
        
        # Plugin metadata
        self._name = "Multinomial Naive Bayes"
        self._description = "Probabilistic classifier for discrete count features, ideal for text classification and document analysis."
        self._category = "Probabilistic Models"
        self._algorithm_type = "Probabilistic Text Classifier"
        self._paper_reference = "McCallum, A., & Nigam, K. (1998). A comparison of event models for naive bayes text classification."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 10
        self._handles_missing_values = False
        self._requires_scaling = False  # Works with raw counts
        self._supports_sparse = True
        self._is_linear = True  # Linear in log space
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._handles_categorical = False
        self._text_specialized = True
        self._count_based = True
        self._probabilistic = True
        self._interpretable = True
        self._fast_training = True
        self._fast_prediction = True
        self._memory_efficient = True
        self._incremental_learning = True
        
        # Internal attributes
        self.model_ = None
        self.vectorizer_ = None
        self.feature_selector_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.feature_count_ = None
        self.class_count_ = None
        self.vocabulary_ = None
        self.feature_importance_scores_ = None
        
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
            "year_introduced": 1995,
            "mathematical_foundation": {
                "bayes_theorem": "P(class|features) ∝ P(features|class) × P(class)",
                "multinomial_assumption": "Features follow multinomial distribution",
                "independence_assumption": "Features are conditionally independent given class",
                "likelihood_calculation": "P(features|class) = ∏ P(feature_i|class)^count_i",
                "smoothing_formula": "P(feature|class) = (count + α) / (total_count + α × vocabulary_size)"
            },
            "key_characteristics": {
                "discrete_features": "Designed for count-based discrete features",
                "text_specialization": "Optimized for text classification tasks",
                "multinomial_distribution": "Models feature counts as multinomial",
                "additive_smoothing": "Laplace/Lidstone smoothing for zero counts",
                "probabilistic_output": "Natural probability estimates",
                "linear_decision_boundary": "Linear in log-probability space"
            },
            "algorithm_mechanics": {
                "training_process": [
                    "Calculate class priors P(class)",
                    "Calculate feature likelihoods P(feature|class)",
                    "Apply smoothing to handle zero counts",
                    "Store log probabilities for numerical stability"
                ],
                "prediction_process": [
                    "Calculate log P(class|features) for each class",
                    "Use log-sum-exp for numerical stability",
                    "Return class with highest posterior probability",
                    "Convert to probabilities using softmax"
                ],
                "smoothing_methods": {
                    "laplace": "Add 1 to all counts (α=1)",
                    "lidstone": "Add α to all counts (generalized Laplace)",
                    "no_smoothing": "α=0, may cause issues with unseen features"
                }
            },
            "strengths": [
                "Extremely fast training and prediction",
                "Works well with small datasets",
                "Naturally handles multi-class problems",
                "Provides interpretable probability estimates",
                "Robust to irrelevant features",
                "Memory efficient",
                "Works well with high-dimensional data",
                "Good performance on text classification",
                "Handles sparse data efficiently",
                "Simple to implement and understand",
                "No hyperparameter tuning required",
                "Incremental learning possible"
            ],
            "weaknesses": [
                "Strong independence assumption often violated",
                "Requires non-negative feature values",
                "Can be outperformed by more complex models",
                "Sensitive to skewed class distributions",
                "May struggle with complex feature interactions",
                "Assumes features follow multinomial distribution",
                "Poor performance if independence assumption is strongly violated",
                "Requires smoothing for unseen features"
            ],
            "ideal_use_cases": [
                "Text classification (spam detection, sentiment analysis)",
                "Document categorization",
                "News article classification",
                "Email filtering",
                "Language detection",
                "Topic modeling",
                "Bag-of-words features",
                "N-gram analysis",
                "Word frequency analysis",
                "Count-based feature vectors",
                "Baseline text classification",
                "Real-time text processing",
                "Large vocabulary problems",
                "Multi-label text classification"
            ],
            "text_processing_integration": {
                "vectorization_methods": {
                    "count_vectorizer": {
                        "description": "Raw term frequencies",
                        "use_case": "Basic text classification",
                        "advantages": ["Simple", "Interpretable", "Fast"],
                        "formula": "count(term, document)"
                    },
                    "tfidf_vectorizer": {
                        "description": "Term frequency-inverse document frequency",
                        "use_case": "Document similarity, information retrieval",
                        "advantages": ["Downweights common terms", "Better for longer documents"],
                        "formula": "tf(t,d) × log(N/df(t))"
                    },
                    "binary_vectorizer": {
                        "description": "Binary term occurrence",
                        "use_case": "Presence/absence of terms",
                        "advantages": ["Reduces effect of term frequency", "Good for short texts"]
                    }
                },
                "preprocessing_options": {
                    "ngrams": "Capture local word dependencies (1-grams to 3-grams)",
                    "stop_words": "Remove common words (the, is, at, etc.)",
                    "lowercase": "Normalize case variations",
                    "min_df": "Remove rare terms",
                    "max_df": "Remove too common terms",
                    "max_features": "Limit vocabulary size"
                }
            },
            "feature_importance_interpretation": {
                "log_probabilities": "Higher values indicate stronger association with class",
                "feature_ranking": "Rank features by log probability differences",
                "class_specific": "Each class has its own feature importance",
                "vocabulary_analysis": "Identify discriminative words/terms"
            },
            "performance_characteristics": {
                "training_complexity": "O(n × d) where n=samples, d=features",
                "prediction_complexity": "O(d × c) where c=classes",
                "memory_usage": "O(d × c) for storing probabilities",
                "scalability": "Excellent for high-dimensional sparse data",
                "incremental_updates": "Can update with new data efficiently"
            },
            "hyperparameter_guide": {
                "alpha": {
                    "range": "0.01 to 10.0",
                    "effect": "Higher values = more smoothing, lower variance",
                    "tuning": "Cross-validation, start with 1.0"
                },
                "vectorization": {
                    "max_features": "10K-100K depending on dataset size",
                    "ngram_range": "(1,1) for simple, (1,2) for better context",
                    "min_df": "2-5 to remove very rare terms",
                    "max_df": "0.8-0.95 to remove very common terms"
                }
            },
            "comparison_with_variants": {
                "vs_gaussian_nb": {
                    "data_type": "Multinomial: discrete counts, Gaussian: continuous",
                    "use_case": "Multinomial: text, Gaussian: sensor data",
                    "assumptions": "Different distribution assumptions"
                },
                "vs_bernoulli_nb": {
                    "features": "Multinomial: count frequency, Bernoulli: binary presence",
                    "text_application": "Multinomial: word counts, Bernoulli: word occurrence",
                    "performance": "Multinomial often better for longer documents"
                },
                "vs_complement_nb": {
                    "class_imbalance": "Complement NB better for imbalanced datasets",
                    "calculation": "Complement uses all classes except target",
                    "use_case": "Complement for skewed data, Multinomial for balanced"
                }
            }
        }
    
    def _create_vectorizer(self):
        """Create text vectorizer based on configuration"""
        if self.vectorization_method == 'count':
            return CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=self.stop_words,
                lowercase=self.lowercase,
                binary=False
            )
        elif self.vectorization_method == 'tfidf':
            return TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=self.stop_words,
                lowercase=self.lowercase,
                use_idf=True,
                smooth_idf=True
            )
        elif self.vectorization_method == 'binary':
            return CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=self.stop_words,
                lowercase=self.lowercase,
                binary=True
            )
        else:
            return CountVectorizer(max_features=self.max_features)
    
    def _validate_features(self, X):
        """Validate that features are non-negative (required for Multinomial NB)"""
        if self.validate_input:
            if np.any(X < 0):
                if self.zero_feature_handling == 'error':
                    raise ValueError("Multinomial Naive Bayes requires non-negative features")
                elif self.zero_feature_handling == 'warn':
                    warnings.warn("Negative features detected. Multinomial NB requires non-negative features.")
                # For 'ignore', just continue
            
            # Check for all-zero samples
            zero_samples = np.sum(X, axis=1) == 0
            if np.any(zero_samples):
                warnings.warn(f"Found {np.sum(zero_samples)} samples with all-zero features")
    
    def fit(self, X, y, 
            sample_weight=None,
            text_data=None):
        """
        Fit the Multinomial Naive Bayes model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features) or list of strings
            Training data (count features) or text data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, optional
            Sample weights
        text_data : bool, optional
            Whether X contains text data that needs vectorization
            
        Returns:
        --------
        self : object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Handle text data
        if text_data or (hasattr(X, 'dtype') and X.dtype == object):
            # X contains text data - needs vectorization
            self.vectorizer_ = self._create_vectorizer()
            X_processed = self.vectorizer_.fit_transform(X)
            self.vocabulary_ = self.vectorizer_.vocabulary_
            self.feature_names_ = self.vectorizer_.get_feature_names_out()
        else:
            # X contains count features already
            X_processed, y = check_X_y(X, y, accept_sparse=True)
            X_processed = X_processed
            if hasattr(X, 'columns'):
                self.feature_names_ = list(X.columns)
            else:
                self.feature_names_ = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        # Validate features
        self._validate_features(X_processed)
        
        # Store training info
        self.n_features_in_ = X_processed.shape[1]
        
        # Encode labels if they're not numeric
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        
        # Feature selection if enabled
        if self.feature_selection and X_processed.shape[1] > self.n_features_to_select:
            from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
            
            self.feature_selector_ = SelectKBest(
                score_func=chi2,  # Good for text features
                k=min(self.n_features_to_select, X_processed.shape[1])
            )
            X_processed = self.feature_selector_.fit_transform(X_processed, y_encoded)
            
            # Update feature names
            if hasattr(self.feature_selector_, 'get_support'):
                selected_features = self.feature_selector_.get_support()
                self.feature_names_ = [name for i, name in enumerate(self.feature_names_) if selected_features[i]]
        
        # Create and fit Multinomial NB model
        self.model_ = MultinomialNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior
        )
        
        self.model_.fit(X_processed, y_encoded, sample_weight=sample_weight)
        
        # Store learned parameters for analysis
        self.class_log_prior_ = self.model_.class_log_prior_
        self.feature_log_prob_ = self.model_.feature_log_prob_
        self.feature_count_ = self.model_.feature_count_
        self.class_count_ = self.model_.class_count_
        
        # Calculate feature importance scores
        self._calculate_feature_importance()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like or list of strings
            Samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self, 'is_fitted_')
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Process features
        X_processed = self._transform_features(X)
        
        # Get predictions
        y_pred_encoded = self.model_.predict(X_processed)
        
        # Convert back to original labels
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like or list of strings
            Samples
            
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Process features
        X_processed = self._transform_features(X)
        
        # Get probability predictions
        probabilities = self.model_.predict_proba(X_processed)
        
        return probabilities
    
    def predict_log_proba(self, X):
        """
        Predict log probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like or list of strings
            Samples
            
        Returns:
        --------
        log_probabilities : array, shape (n_samples, n_classes)
            Log probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X_processed = self._transform_features(X)
        return self.model_.predict_log_proba(X_processed)
    
    def _transform_features(self, X):
        """Transform features using fitted vectorizer and feature selector"""
        # Handle text data
        if self.vectorizer_ is not None:
            X_processed = self.vectorizer_.transform(X)
        else:
            X_processed = check_array(X, accept_sparse=True)
        
        # Apply feature selection if fitted
        if self.feature_selector_ is not None:
            X_processed = self.feature_selector_.transform(X_processed)
        
        return X_processed
    
    def _calculate_feature_importance(self):
        """Calculate feature importance based on log probabilities"""
        if self.feature_log_prob_ is None:
            return
        
        # Calculate feature importance as the maximum absolute log probability difference
        # between classes for each feature
        n_classes, n_features = self.feature_log_prob_.shape
        importance_scores = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            feature_log_probs = self.feature_log_prob_[:, feature_idx]
            # Use range (max - min) as importance measure
            importance_scores[feature_idx] = np.max(feature_log_probs) - np.min(feature_log_probs)
        
        self.feature_importance_scores_ = importance_scores
    
    def get_feature_importance(self, top_k=None):
        """
        Get feature importance scores
        
        Parameters:
        -----------
        top_k : int, optional
            Return only top k features
            
        Returns:
        --------
        importance_info : dict
            Feature importance information
        """
        if not self.is_fitted_ or self.feature_importance_scores_ is None:
            return None
        
        importance_scores = self.feature_importance_scores_.copy()
        
        # Get top features
        if top_k is not None:
            top_indices = np.argsort(importance_scores)[::-1][:top_k]
        else:
            top_indices = np.argsort(importance_scores)[::-1]
        
        return {
            "feature_names": [self.feature_names_[i] for i in top_indices],
            "importance_scores": importance_scores[top_indices],
            "feature_indices": top_indices,
            "total_features": len(self.feature_names_)
        }
    
    def get_class_feature_analysis(self, class_name=None, top_k=20):
        """
        Get detailed feature analysis for specific class
        
        Parameters:
        -----------
        class_name : str, optional
            Class to analyze (if None, analyze all classes)
        top_k : int, default=20
            Number of top features to return
            
        Returns:
        --------
        analysis_info : dict
            Detailed class-specific feature analysis
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {}
        
        if class_name is not None:
            # Analyze specific class
            if class_name not in self.classes_:
                return {"error": f"Class '{class_name}' not found"}
            
            class_idx = np.where(self.classes_ == class_name)[0][0]
            feature_log_probs = self.feature_log_prob_[class_idx]
            
            # Get top features for this class
            top_indices = np.argsort(feature_log_probs)[::-1][:top_k]
            
            analysis[class_name] = {
                "top_features": [self.feature_names_[i] for i in top_indices],
                "log_probabilities": feature_log_probs[top_indices],
                "probabilities": np.exp(feature_log_probs[top_indices]),
                "feature_indices": top_indices
            }
        else:
            # Analyze all classes
            for class_idx, class_name in enumerate(self.classes_):
                feature_log_probs = self.feature_log_prob_[class_idx]
                top_indices = np.argsort(feature_log_probs)[::-1][:top_k]
                
                analysis[class_name] = {
                    "top_features": [self.feature_names_[i] for i in top_indices],
                    "log_probabilities": feature_log_probs[top_indices],
                    "probabilities": np.exp(feature_log_probs[top_indices]),
                    "feature_indices": top_indices
                }
        
        return analysis
    
    def get_vocabulary_analysis(self):
        """
        Get vocabulary and text processing analysis
        
        Returns:
        --------
        vocab_info : dict
            Vocabulary analysis information
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "vectorization_method": self.vectorization_method,
            "total_features": self.n_features_in_,
            "vocabulary_size": len(self.vocabulary_) if self.vocabulary_ else self.n_features_in_,
            "ngram_range": self.ngram_range,
            "feature_selection_applied": self.feature_selector_ is not None
        }
        
        if self.vectorizer_:
            analysis.update({
                "max_features_limit": self.max_features,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "stop_words_used": self.stop_words is not None,
                "lowercase_applied": self.lowercase
            })
        
        if self.feature_selector_:
            analysis.update({
                "features_before_selection": len(self.feature_names_) if hasattr(self, 'original_feature_names_') else "Unknown",
                "features_after_selection": self.n_features_in_,
                "feature_selection_method": "Chi-square"
            })
        
        return analysis
    
    def plot_feature_importance(self, top_k=20, figsize=(12, 8)):
        """
        Plot feature importance
        
        Parameters:
        -----------
        top_k : int, default=20
            Number of top features to display
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Feature importance plot
        """
        importance_info = self.get_feature_importance(top_k=top_k)
        if not importance_info:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        feature_names = importance_info["feature_names"]
        scores = importance_info["importance_scores"]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in feature_names])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance Score (Log Probability Range)')
        ax.set_title(f'Top {len(feature_names)} Most Important Features - Multinomial Naive Bayes')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_class_feature_analysis(self, top_k=15, figsize=(15, 10)):
        """
        Plot class-specific feature analysis
        
        Parameters:
        -----------
        top_k : int, default=15
            Number of top features per class
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Class feature analysis plot
        """
        if not self.is_fitted_:
            return None
        
        n_classes = len(self.classes_)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_classes == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        class_analysis = self.get_class_feature_analysis(top_k=top_k)
        
        for i, (class_name, class_data) in enumerate(class_analysis.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            feature_names = class_data["top_features"]
            log_probs = class_data["log_probabilities"]
            
            y_pos = np.arange(len(feature_names))
            bars = ax.barh(y_pos, log_probs, color=plt.cm.Set3(i), alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in feature_names])
            ax.invert_yaxis()
            ax.set_xlabel('Log Probability')
            ax.set_title(f'Top Features for Class: {class_name}')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, prob in zip(bars, log_probs):
                width = bar.get_width()
                ax.text(width + (max(log_probs) - min(log_probs)) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{prob:.2f}', ha='left', va='center', fontsize=7)
        
        # Hide empty subplots
        for i in range(n_classes, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_probability_distribution(self, X_sample, figsize=(12, 8)):
        """
        Plot probability distribution for sample predictions
        
        Parameters:
        -----------
        X_sample : array-like
            Sample data to analyze
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Probability distribution plot
        """
        if not self.is_fitted_:
            return None
        
        # Get probabilities
        probabilities = self.predict_proba(X_sample)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Box plot of probabilities by class
        prob_data = []
        class_labels = []
        for i, class_name in enumerate(self.classes_):
            prob_data.append(probabilities[:, i])
            class_labels.append(str(class_name))
        
        ax1.boxplot(prob_data, labels=class_labels)
        ax1.set_ylabel('Predicted Probability')
        ax1.set_title('Distribution of Predicted Probabilities by Class')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of maximum probabilities
        max_probs = np.max(probabilities, axis=1)
        ax2.hist(max_probs, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probs):.3f}')
        ax2.set_xlabel('Maximum Predicted Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Prediction Confidence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 📊 Multinomial Naive Bayes Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Text Processing", "Advanced", "Analysis", "Info"])
        
        with tab1:
            st.markdown("**Core Naive Bayes Parameters**")
            
            # Alpha (smoothing)
            alpha = st.number_input(
                "Smoothing Parameter (α):",
                value=float(self.alpha),
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                help="Additive smoothing parameter. 0=no smoothing, 1=Laplace smoothing",
                key=f"{key_prefix}_alpha"
            )
            
            # Fit prior
            fit_prior = st.checkbox(
                "Learn Class Priors",
                value=self.fit_prior,
                help="Whether to learn class prior probabilities from data",
                key=f"{key_prefix}_fit_prior"
            )
            
            # Validation
            validate_input = st.checkbox(
                "Validate Input Features",
                value=self.validate_input,
                help="Check that all features are non-negative",
                key=f"{key_prefix}_validate_input"
            )
            
            if validate_input:
                zero_feature_handling = st.selectbox(
                    "Negative Feature Handling:",
                    options=['warn', 'error', 'ignore'],
                    index=['warn', 'error', 'ignore'].index(self.zero_feature_handling),
                    help="How to handle negative features",
                    key=f"{key_prefix}_zero_feature_handling"
                )
            else:
                zero_feature_handling = 'ignore'
        
        with tab2:
            st.markdown("**Text Vectorization Settings**")
            
            # Vectorization method
            vectorization_method = st.selectbox(
                "Vectorization Method:",
                options=['count', 'tfidf', 'binary'],
                index=['count', 'tfidf', 'binary'].index(self.vectorization_method),
                help="count: raw frequencies, tfidf: weighted frequencies, binary: presence/absence",
                key=f"{key_prefix}_vectorization_method"
            )
            
            # Max features
            max_features = st.number_input(
                "Max Features:",
                value=int(self.max_features),
                min_value=100,
                max_value=100000,
                step=1000,
                help="Maximum number of features to extract",
                key=f"{key_prefix}_max_features"
            )
            
            # N-gram range
            ngram_min = st.number_input(
                "N-gram Min:",
                value=int(self.ngram_range[0]),
                min_value=1,
                max_value=3,
                help="Minimum n-gram size",
                key=f"{key_prefix}_ngram_min"
            )
            
            ngram_max = st.number_input(
                "N-gram Max:",
                value=int(self.ngram_range[1]),
                min_value=ngram_min,
                max_value=5,
                help="Maximum n-gram size",
                key=f"{key_prefix}_ngram_max"
            )
            
            ngram_range = (ngram_min, ngram_max)
            
            # Document frequency filtering
            min_df = st.number_input(
                "Min Document Frequency:",
                value=int(self.min_df) if isinstance(self.min_df, int) else float(self.min_df),
                min_value=1,
                max_value=100,
                help="Ignore terms with document frequency lower than threshold",
                key=f"{key_prefix}_min_df"
            )
            
            max_df = st.slider(
                "Max Document Frequency:",
                min_value=0.5,
                max_value=1.0,
                value=float(self.max_df),
                step=0.05,
                help="Ignore terms with document frequency higher than threshold",
                key=f"{key_prefix}_max_df"
            )
            
            # Text preprocessing
            lowercase = st.checkbox(
                "Convert to Lowercase",
                value=self.lowercase,
                help="Convert all text to lowercase",
                key=f"{key_prefix}_lowercase"
            )
            
            # Stop words
            stop_words_option = st.selectbox(
                "Stop Words:",
                options=['None', 'english'],
                index=0 if self.stop_words is None else 1,
                help="Remove common words",
                key=f"{key_prefix}_stop_words_option"
            )
            
            stop_words = None if stop_words_option == 'None' else 'english'
        
        with tab3:
            st.markdown("**Advanced Settings**")
            
            # Feature selection
            feature_selection = st.checkbox(
                "Enable Feature Selection",
                value=self.feature_selection,
                help="Use chi-square test to select most informative features",
                key=f"{key_prefix}_feature_selection"
            )
            
            if feature_selection:
                n_features_to_select = st.number_input(
                    "Features to Select:",
                    value=int(self.n_features_to_select),
                    min_value=100,
                    max_value=50000,
                    step=100,
                    help="Number of top features to keep",
                    key=f"{key_prefix}_n_features_to_select"
                )
            else:
                n_features_to_select = self.n_features_to_select
            
            # Smoothing method info
            smoothing_method = st.selectbox(
                "Smoothing Method:",
                options=['laplace', 'lidstone'],
                index=['laplace', 'lidstone'].index(self.smoothing_method),
                help="laplace: α=1, lidstone: custom α",
                key=f"{key_prefix}_smoothing_method"
            )
            
            # Log probability thresholding
            feature_log_prob_thresholding = st.checkbox(
                "Log Probability Thresholding",
                value=self.feature_log_prob_thresholding,
                help="Apply threshold to feature log probabilities",
                key=f"{key_prefix}_feature_log_prob_thresholding"
            )
            
            if feature_log_prob_thresholding:
                log_prob_threshold = st.number_input(
                    "Log Probability Threshold:",
                    value=float(self.log_prob_threshold),
                    min_value=-20.0,
                    max_value=-1.0,
                    step=1.0,
                    help="Minimum log probability value",
                    key=f"{key_prefix}_log_prob_threshold"
                )
            else:
                log_prob_threshold = self.log_prob_threshold
        
        with tab4:
            st.markdown("**Analysis Options**")
            
            # Feature importance analysis
            show_feature_importance = st.checkbox(
                "Feature Importance Analysis",
                value=True,
                help="Analyze most important features",
                key=f"{key_prefix}_show_feature_importance"
            )
            
            # Class-specific analysis
            show_class_analysis = st.checkbox(
                "Class-Specific Feature Analysis",
                value=True,
                help="Show top features for each class",
                key=f"{key_prefix}_show_class_analysis"
            )
            
            # Vocabulary analysis
            show_vocabulary_analysis = st.checkbox(
                "Vocabulary Analysis",
                value=True,
                help="Analyze text processing results",
                key=f"{key_prefix}_show_vocabulary_analysis"
            )
            
            # Probability analysis
            show_probability_analysis = st.checkbox(
                "Probability Distribution Analysis",
                value=True,
                help="Analyze prediction probabilities",
                key=f"{key_prefix}_show_probability_analysis"
            )
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if SKLEARN_AVAILABLE:
                st.success("✅ scikit-learn is available")
            else:
                st.error("❌ scikit-learn not installed. Run: pip install scikit-learn")
            
            st.info("""
            **Multinomial Naive Bayes** - Text Classification Specialist:
            • 📊 Designed for discrete count features
            • 📝 Ideal for text classification tasks
            • 🚀 Extremely fast training and prediction
            • 📈 Natural probability estimates
            • 🎯 Works well with high-dimensional data
            • 💾 Memory efficient for sparse features
            
            **Perfect for:**
            • Text classification
            • Document categorization
            • Spam detection
            • Sentiment analysis
            • N-gram analysis
            """)
            
            # Mathematical foundation
            if st.button("🧮 Mathematical Foundation", key=f"{key_prefix}_math_foundation"):
                st.markdown("""
                **Bayes' Theorem:**
                P(class|features) ∝ P(features|class) × P(class)
                
                **Multinomial Likelihood:**
                P(features|class) = ∏ P(feature_i|class)^count_i
                
                **Smoothing Formula:**
                P(feature|class) = (count + α) / (total_count + α × vocab_size)
                
                **Independence Assumption:**
                Features are conditionally independent given the class
                """)
            
            # Text processing guide
            if st.button("📝 Text Processing Guide", key=f"{key_prefix}_text_guide"):
                st.markdown("""
                **Vectorization Methods:**
                
                **Count Vectorizer:**
                - Raw term frequencies
                - Good for basic classification
                - Interpretable results
                
                **TF-IDF Vectorizer:**
                - Weighted by inverse document frequency
                - Reduces impact of common words
                - Better for document similarity
                
                **Binary Vectorizer:**
                - Presence/absence of terms
                - Good for short texts
                - Reduces impact of term frequency
                """)
            
            # Feature importance guide
            if st.button("🎯 Feature Interpretation", key=f"{key_prefix}_feature_guide"):
                st.markdown("""
                **Understanding Feature Importance:**
                
                **Log Probabilities:**
                - Higher values = stronger association with class
                - Negative values are normal (log of probabilities)
                - Compare relative differences between classes
                
                **Class-Specific Features:**
                - Each class has its own "vocabulary"
                - Top features are most discriminative
                - Useful for understanding model decisions
                
                **Vocabulary Analysis:**
                - Shows text processing effectiveness
                - Helps tune preprocessing parameters
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "alpha": alpha,
            "fit_prior": fit_prior,
            "vectorization_method": vectorization_method,
            "max_features": max_features,
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
            "stop_words": stop_words,
            "lowercase": lowercase,
            "feature_selection": feature_selection,
            "n_features_to_select": n_features_to_select,
            "smoothing_method": smoothing_method,
            "feature_log_prob_thresholding": feature_log_prob_thresholding,
            "log_prob_threshold": log_prob_threshold,
            "validate_input": validate_input,
            "zero_feature_handling": zero_feature_handling,
            "_ui_options": {
                "show_feature_importance": show_feature_importance,
                "show_class_analysis": show_class_analysis,
                "show_vocabulary_analysis": show_vocabulary_analysis,
                "show_probability_analysis": show_probability_analysis
            }
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return MultinomialNaiveBayesPlugin(
            alpha=hyperparameters.get("alpha", self.alpha),
            fit_prior=hyperparameters.get("fit_prior", self.fit_prior),
            vectorization_method=hyperparameters.get("vectorization_method", self.vectorization_method),
            max_features=hyperparameters.get("max_features", self.max_features),
            ngram_range=hyperparameters.get("ngram_range", self.ngram_range),
            min_df=hyperparameters.get("min_df", self.min_df),
            max_df=hyperparameters.get("max_df", self.max_df),
            stop_words=hyperparameters.get("stop_words", self.stop_words),
            lowercase=hyperparameters.get("lowercase", self.lowercase),
            feature_selection=hyperparameters.get("feature_selection", self.feature_selection),
            n_features_to_select=hyperparameters.get("n_features_to_select", self.n_features_to_select),
            smoothing_method=hyperparameters.get("smoothing_method", self.smoothing_method),
            feature_log_prob_thresholding=hyperparameters.get("feature_log_prob_thresholding", self.feature_log_prob_thresholding),
            log_prob_threshold=hyperparameters.get("log_prob_threshold", self.log_prob_threshold),
            validate_input=hyperparameters.get("validate_input", self.validate_input),
            zero_feature_handling=hyperparameters.get("zero_feature_handling", self.zero_feature_handling)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for Multinomial Naive Bayes
        
        Multinomial NB works directly with count features, so minimal preprocessing needed.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # Multinomial NB requires non-negative features
        if self.validate_input and np.any(X_processed < 0):
            warnings.warn("Multinomial Naive Bayes requires non-negative features.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if Multinomial Naive Bayes is compatible with the given data
        
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
            return False, f"Multinomial NB requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for text data (object dtype)
        if hasattr(X, 'dtype') and X.dtype == object:
            return True, "Text data detected - Multinomial NB is ideal for text classification"
        
        # Check for non-negative features
        if np.any(X < 0):
            return False, "Multinomial NB requires non-negative features (counts). Consider using Gaussian NB for continuous features."
        
        # Check for sparse or count-like data
        if hasattr(X, 'nnz'):  # Sparse matrix
            return True, "Sparse data detected - Multinomial NB works well with sparse count features"
        
        # Check if data looks like counts (integers)
        if np.all(X >= 0) and np.all(X == X.astype(int)):
            return True, "Count-based features detected - perfect for Multinomial NB"
        
        # Check if data could be normalized counts
        if np.all(X >= 0) and np.all(X <= 1):
            return True, "Non-negative features detected - compatible with Multinomial NB (consider if these represent counts or frequencies)"
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
        
        return True, "Multinomial NB is compatible with this data"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_[:10] if self.feature_names_ else None,  # Show first 10
            "total_features": len(self.feature_names_) if self.feature_names_ else None,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "alpha": self.alpha,
            "fit_prior": self.fit_prior,
            "vectorization_applied": self.vectorizer_ is not None,
            "vectorization_method": self.vectorization_method if self.vectorizer_ else None,
            "feature_selection_applied": self.feature_selector_ is not None,
            "vocabulary_size": len(self.vocabulary_) if self.vocabulary_ else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Multinomial Naive Bayes",
            "training_completed": True,
            "naive_bayes_characteristics": {
                "probabilistic": True,
                "assumes_independence": True,
                "multinomial_distribution": True,
                "discrete_features": True,
                "fast_training": True,
                "fast_prediction": True,
                "incremental_learning": True
            },
            "model_configuration": {
                "smoothing_parameter": self.alpha,
                "learns_priors": self.fit_prior,
                "feature_validation": self.validate_input,
                "text_processing": self.vectorizer_ is not None
            },
            "text_processing_info": self.get_vocabulary_analysis(),
            "feature_analysis": {
                "total_features": self.n_features_in_,
                "feature_selection_used": self.feature_selector_ is not None,
                "feature_importance_available": self.feature_importance_scores_ is not None
            },
            "mathematical_properties": {
                "bayes_theorem": "P(class|features) ∝ P(features|class) × P(class)",
                "independence_assumption": "P(features|class) = ∏ P(feature_i|class)^count_i",
                "smoothing_formula": "P(feature|class) = (count + α) / (total + α × vocab_size)",
                "logarithmic_computation": "Uses log probabilities for numerical stability"
            },
            "performance_characteristics": {
                "training_complexity": "O(n × d × c)",
                "prediction_complexity": "O(d × c)",
                "memory_usage": "O(d × c)",
                "scalability": "Excellent for high-dimensional sparse data",
                "suitable_for_streaming": True
            }
        }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for Multinomial Naive Bayes.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Required for metrics like McFadden's R-squared.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics.
        y_proba : np.ndarray, optional
            Predicted probabilities. Required for metrics like McFadden's R-squared.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not self.model_:
            return {"error": "Model not fitted. Cannot retrieve Multinomial Naive Bayes specific metrics."}

        metrics = {}
        prefix = "mnb_" # Multinomial Naive Bayes

        # Smoothing parameter
        metrics[f"{prefix}alpha_smoothing_param"] = self.alpha

        # Vocabulary size (if text processing was done)
        if self.vectorizer_ and hasattr(self.vectorizer_, 'vocabulary_') and self.vectorizer_.vocabulary_:
            metrics[f"{prefix}vocabulary_size"] = len(self.vectorizer_.vocabulary_)
        elif self.vocabulary_: # Fallback if vocabulary_ was set directly
             metrics[f"{prefix}vocabulary_size"] = len(self.vocabulary_)


        # Number of features
        metrics[f"{prefix}num_features_in_model"] = self.model_.n_features_in_ # Features seen by the actual NB model
        if self.feature_selector_ and hasattr(self.feature_selector_, 'get_support'):
            metrics[f"{prefix}num_features_before_selection"] = len(self.feature_selector_.get_support())
            metrics[f"{prefix}num_features_after_selection"] = int(np.sum(self.feature_selector_.get_support()))
        else:
            metrics[f"{prefix}num_features_total"] = self.n_features_in_ # Initial features before any selection

        # Class log priors
        if self.class_log_prior_ is not None:
            metrics[f"{prefix}mean_class_log_prior"] = float(np.mean(self.class_log_prior_))
            metrics[f"{prefix}std_class_log_prior"] = float(np.std(self.class_log_prior_))
            for i, class_label in enumerate(self.classes_):
                 metrics[f"{prefix}class_log_prior_{class_label}"] = float(self.class_log_prior_[i])


        # Feature importance scores (derived from feature_log_prob_)
        if self.feature_importance_scores_ is not None:
            metrics[f"{prefix}mean_feature_importance"] = float(np.mean(self.feature_importance_scores_))
            metrics[f"{prefix}std_feature_importance"] = float(np.std(self.feature_importance_scores_))
            metrics[f"{prefix}max_feature_importance"] = float(np.max(self.feature_importance_scores_))
        
        # Number of classes
        if self.classes_ is not None:
            metrics[f"{prefix}num_classes"] = len(self.classes_)


        # McFadden's Pseudo R-squared (requires y_true and y_proba)
        if y_true is not None and y_proba is not None and self.label_encoder_ is not None and self.classes_ is not None:
            try:
                y_true_encoded = self.label_encoder_.transform(y_true)
                n_samples = len(y_true_encoded)
                n_classes_model = len(self.classes_)

                # Log-likelihood of the full model
                clipped_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
                log_likelihoods_model = np.log(clipped_proba[np.arange(n_samples), y_true_encoded])
                ll_model = np.sum(log_likelihoods_model)

                # Log-likelihood of the null model (intercept-only)
                class_counts = np.bincount(y_true_encoded, minlength=n_classes_model)
                class_probas_null = class_counts / n_samples
                
                ll_null = 0
                for k_idx in range(n_classes_model):
                    if class_counts[k_idx] > 0 and class_probas_null[k_idx] > 0: # Ensure proba > 0 before log
                         ll_null += class_counts[k_idx] * np.log(np.clip(class_probas_null[k_idx], 1e-15, 1))
                
                if ll_null == 0:
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = 1.0 if ll_model == 0 else 0.0
                elif ll_model > ll_null: 
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = 0.0
                else:
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = float(1 - (ll_model / ll_null))
            except Exception as e:
                metrics[f"{prefix}mcfaddens_pseudo_r2_error"] = str(e)

        if not metrics:
            metrics['info'] = "No specific Multinomial Naive Bayes metrics were available (e.g., model not fitted properly or y_true/y_proba not provided for Pseudo R2)."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return MultinomialNaiveBayesPlugin()