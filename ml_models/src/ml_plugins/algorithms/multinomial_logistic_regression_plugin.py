import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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


class MultinomialLogisticRegressionPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Multinomial Logistic Regression Plugin - Native Multi-class Classification
    
    This plugin implements true multinomial logistic regression that naturally handles
    multiple classes in a single unified model, using the multinomial (softmax) 
    distribution for probability estimation.
    
    Key Features:
    - Native multi-class support (not binary decomposition)
    - Softmax probability estimation
    - L1, L2, and Elastic Net regularization
    - Feature importance analysis
    - Coefficient interpretation per class
    - Advanced solver options
    - Automatic feature scaling
    - Cross-validation performance estimation
    """
    
    def __init__(
        self,
        # Core multinomial parameters
        C=1.0,
        penalty='l2',
        l1_ratio=None,
        solver='lbfgs',
        max_iter=1000,
        tol=1e-4,
        
        # Multi-class strategy (enforce multinomial)
        multi_class='multinomial',
        
        # Feature preprocessing
        auto_scale_features=True,
        
        # Advanced options
        class_weight=None,
        fit_intercept=True,
        intercept_scaling=1.0,
        warm_start=False,
        
        # Performance options
        n_jobs=None,
        verbose=0,
        random_state=42
    ):
        super().__init__()
        
        # Core parameters
        self.C = C
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.multi_class = multi_class  # Always 'multinomial' for this plugin
        
        # Feature preprocessing
        self.auto_scale_features = auto_scale_features
        
        # Advanced options
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.warm_start = warm_start
        
        # Performance options
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "Multinomial Logistic Regression"
        self._description = "Native multi-class logistic regression using multinomial (softmax) distribution"
        self._category = "Linear Models"
        
        # Required capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._min_samples_required = 20
        
        # Internal state
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.is_fitted_ = False
        self.feature_names_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        
        # Performance tracking
        self.training_score_ = None
        self.cv_scores_ = None
        
        # Multinomial analysis
        self.multinomial_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Multinomial Logistic Regression model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample
        
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        # 🎯 STORE FEATURE NAMES BEFORE VALIDATION!
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Store classes and validate multi-class
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        # Label encoding for string labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        
        # Feature scaling if requested
        if self.auto_scale_features:
            self.scaler_ = StandardScaler()
            X_processed = self.scaler_.fit_transform(X)
        else:
            X_processed = X.copy()
            self.scaler_ = None
        
        # Ensure multinomial multi-class (this is what makes it native multi-class)
        multi_class = 'multinomial'
        
        # Choose appropriate solver for multinomial
        solver = self._get_optimal_solver()
        
        # Create and configure the multinomial logistic regression model
        self.model_ = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            l1_ratio=self.l1_ratio if self.penalty == 'elasticnet' else None,
            solver=solver,
            max_iter=self.max_iter,
            tol=self.tol,
            multi_class=multi_class,  # Key: Force multinomial approach
            class_weight=self.class_weight,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Fit the model
        if sample_weight is not None:
            self.model_.fit(X_processed, y_encoded, sample_weight=sample_weight)
        else:
            self.model_.fit(X_processed, y_encoded)
        
        # Store training performance
        self.training_score_ = self.model_.score(X_processed, y_encoded)
        
        # Perform cross-validation for robust performance estimation
        try:
            self.cv_scores_ = cross_val_score(
                self.model_, X_processed, y_encoded, 
                cv=min(5, self.n_classes_), 
                scoring='accuracy'
            )
        except Exception:
            self.cv_scores_ = None
        
        # Perform multinomial analysis
        self._analyze_multinomial_model(X_processed, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        self._check_fitted()
        X = check_array(X)
        
        # Apply same preprocessing as training
        if self.scaler_ is not None:
            X_processed = self.scaler_.transform(X)
        else:
            X_processed = X
        
        # Get predictions and decode labels
        y_pred_encoded = self.model_.predict(X_processed)
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        self._check_fitted()
        X = check_array(X)
        
        # Apply same preprocessing as training
        if self.scaler_ is not None:
            X_processed = self.scaler_.transform(X)
        else:
            X_processed = X
        
        # Get multinomial probabilities
        probabilities = self.model_.predict_proba(X_processed)
        
        return probabilities
    
    def decision_function(self, X):
        """Compute decision function for samples in X"""
        self._check_fitted()
        X = check_array(X)
        
        # Apply same preprocessing as training
        if self.scaler_ is not None:
            X_processed = self.scaler_.transform(X)
        else:
            X_processed = X
        
        # Get decision scores
        decision_scores = self.model_.decision_function(X_processed)
        
        return decision_scores
    
    def _get_optimal_solver(self):
        """Choose optimal solver based on problem characteristics and penalty"""
        
        # For multinomial logistic regression, solver choice is critical
        if self.penalty == 'elasticnet':
            return 'saga'  # Only solver supporting elasticnet
        elif self.penalty == 'l1':
            return 'saga'  # Best for L1 with multinomial
        elif self.penalty == 'l2':
            if self.n_features_in_ > 1000:
                return 'sag'  # Fast for large datasets
            else:
                return 'lbfgs'  # Best for small-medium datasets
        elif self.penalty == 'none':
            return 'lbfgs'  # Good for unregularized problems
        else:
            return 'lbfgs'  # Default fallback
    
    def _check_fitted(self):
        """Check if model is fitted"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
    
    def _analyze_multinomial_model(self, X, y):
        """Analyze the fitted multinomial model"""
        
        # Basic model characteristics
        self.multinomial_analysis_ = {
            'model_type': 'Multinomial Logistic Regression',
            'native_multiclass': True,
            'probability_model': 'Softmax (Multinomial)',
            'n_classes': self.n_classes_,
            'n_features': self.n_features_in_,
            'solver_used': self.model_.solver,
            'penalty_used': self.model_.penalty,
            'convergence': {
                'converged': self.model_.n_iter_ < self.max_iter,
                'iterations': self.model_.n_iter_[0] if hasattr(self.model_.n_iter_, '__len__') else self.model_.n_iter_,
                'max_iterations': self.max_iter
            }
        }
        
        # Coefficient analysis
        if hasattr(self.model_, 'coef_'):
            coefficients = self.model_.coef_  # Shape: (n_classes, n_features)
            
            # Per-class coefficient analysis
            coef_analysis = {}
            for i, class_name in enumerate(self.classes_):
                class_coefs = coefficients[i] if self.n_classes_ > 2 else coefficients[0]
                
                coef_analysis[str(class_name)] = {
                    'max_coefficient': float(np.max(np.abs(class_coefs))),
                    'mean_abs_coefficient': float(np.mean(np.abs(class_coefs))),
                    'n_positive_coefs': int(np.sum(class_coefs > 0)),
                    'n_negative_coefs': int(np.sum(class_coefs < 0)),
                    'coefficient_range': [float(np.min(class_coefs)), float(np.max(class_coefs))]
                }
            
            self.multinomial_analysis_['coefficient_analysis'] = coef_analysis
            
            # Feature importance (average absolute coefficients across classes)
            if self.n_classes_ > 2:
                avg_abs_coefs = np.mean(np.abs(coefficients), axis=0)
            else:
                avg_abs_coefs = np.abs(coefficients[0])
            
            self.multinomial_analysis_['feature_importance'] = {
                'importance_scores': avg_abs_coefs.tolist(),
                'top_features': {
                    'indices': np.argsort(avg_abs_coefs)[-10:][::-1].tolist(),
                    'names': [self.feature_names_[i] for i in np.argsort(avg_abs_coefs)[-10:][::-1]],
                    'scores': avg_abs_coefs[np.argsort(avg_abs_coefs)[-10:][::-1]].tolist()
                }
            }
        
        # Intercept analysis
        if hasattr(self.model_, 'intercept_') and self.fit_intercept:
            intercepts = self.model_.intercept_
            
            intercept_analysis = {}
            for i, class_name in enumerate(self.classes_):
                intercept_val = intercepts[i] if self.n_classes_ > 2 else intercepts[0]
                intercept_analysis[str(class_name)] = float(intercept_val)
            
            self.multinomial_analysis_['intercept_analysis'] = intercept_analysis
        
        # Probability calibration analysis
        if len(X) > 50:  # Only for reasonably sized datasets
            # Sample some predictions for calibration analysis
            sample_idx = np.random.choice(len(X), min(100, len(X)), replace=False)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            
            # Get probabilities and predictions
            probas = self.model_.predict_proba(X_sample)
            preds = self.model_.predict(X_sample)
            
            # Analyze probability quality
            max_probas = np.max(probas, axis=1)
            correct_preds = (preds == y_sample)
            
            self.multinomial_analysis_['probability_analysis'] = {
                'mean_max_probability': float(np.mean(max_probas)),
                'std_max_probability': float(np.std(max_probas)),
                'correct_high_confidence': float(np.mean(correct_preds[max_probas > 0.8])) if np.sum(max_probas > 0.8) > 0 else 0.0,
                'incorrect_high_confidence': float(np.mean(~correct_preds[max_probas > 0.8])) if np.sum(max_probas > 0.8) > 0 else 0.0,
                'probability_distribution': {
                    'min': float(np.min(max_probas)),
                    'max': float(np.max(max_probas)),
                    'median': float(np.median(max_probas)),
                    'q25': float(np.percentile(max_probas, 25)),
                    'q75': float(np.percentile(max_probas, 75))
                }
            }
        
        # Regularization effect analysis
        if self.penalty != 'none':
            self.multinomial_analysis_['regularization_analysis'] = {
                'penalty_type': self.penalty,
                'C_value': self.C,
                'regularization_strength': 1.0 / self.C,
                'l1_ratio': self.l1_ratio if self.penalty == 'elasticnet' else None,
                'sparsity_level': self._calculate_sparsity() if hasattr(self.model_, 'coef_') else None
            }
    
    def _calculate_sparsity(self):
        """Calculate sparsity level of coefficients"""
        if not hasattr(self.model_, 'coef_'):
            return None
        
        coefficients = self.model_.coef_
        total_coefs = coefficients.size
        zero_coefs = np.sum(np.abs(coefficients) < 1e-6)
        sparsity = zero_coefs / total_coefs
        
        return float(sparsity)
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """
        Get feature importance based on average absolute coefficients across classes
        
        Returns:
        --------
        dict : Feature importance information including per-class and aggregated importance
        """
        if not self.is_fitted_ or not hasattr(self.model_, 'coef_'):
            return None
        
        coefficients = self.model_.coef_
        
        # Per-class feature importance
        per_class_importance = {}
        for i, class_name in enumerate(self.classes_):
            if self.n_classes_ > 2:
                class_coefs = np.abs(coefficients[i])
            else:
                class_coefs = np.abs(coefficients[0])
            
            per_class_importance[str(class_name)] = {
                'importance': class_coefs.tolist(),
                'feature_names': self.feature_names_,
                'max_importance': float(np.max(class_coefs)),
                'mean_importance': float(np.mean(class_coefs))
            }
        
        # Aggregated importance (mean absolute coefficient across classes)
        if self.n_classes_ > 2:
            aggregated_importance = np.mean(np.abs(coefficients), axis=0)
        else:
            aggregated_importance = np.abs(coefficients[0])
        
        # Sort features by importance
        importance_order = np.argsort(aggregated_importance)[::-1]
        
        return {
            'per_class': per_class_importance,
            'aggregated': {
                'importance': aggregated_importance.tolist(),
                'feature_names': self.feature_names_,
                'sorted_indices': importance_order.tolist(),
                'sorted_features': [self.feature_names_[i] for i in importance_order],
                'sorted_importance': aggregated_importance[importance_order].tolist()
            },
            'method': 'Absolute Coefficients (Multinomial Logistic Regression)',
            'interpretation': 'Higher values indicate stronger linear relationship with log-odds'
        }
    
    def get_multinomial_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of the multinomial model"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        return self.multinomial_analysis_.copy()
    
    def plot_coefficients(self, max_features=20, figsize=(12, 8)):
        """
        Plot coefficient heatmap showing how each feature affects each class
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features to display
        figsize : tuple
            Figure size for the plot
        """
        if not self.is_fitted_ or not hasattr(self.model_, 'coef_'):
            st.warning("Model must be fitted to plot coefficients")
            return
        
        coefficients = self.model_.coef_
        
        # Handle binary vs multiclass
        if self.n_classes_ == 2:
            # For binary classification, create symmetric matrix
            coef_matrix = np.vstack([-coefficients[0], coefficients[0]])
            class_labels = [f"Not {self.classes_[1]}", str(self.classes_[1])]
        else:
            coef_matrix = coefficients
            class_labels = [str(c) for c in self.classes_]
        
        # Select top features by importance
        if len(self.feature_names_) > max_features:
            feature_importance = np.mean(np.abs(coef_matrix), axis=0)
            top_feature_indices = np.argsort(feature_importance)[-max_features:]
            coef_matrix = coef_matrix[:, top_feature_indices]
            feature_names = [self.feature_names_[i] for i in top_feature_indices]
        else:
            feature_names = self.feature_names_
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use diverging colormap centered at zero
        max_abs_coef = np.max(np.abs(coef_matrix))
        
        sns.heatmap(
            coef_matrix,
            xticklabels=feature_names,
            yticklabels=class_labels,
            annot=True,
            fmt='.3f',
            center=0,
            cmap='RdBu_r',
            vmin=-max_abs_coef,
            vmax=max_abs_coef,
            cbar_kws={'label': 'Coefficient Value'},
            ax=ax
        )
        
        ax.set_title(f'Multinomial Logistic Regression Coefficients\n'
                    f'({len(feature_names)} features, {self.n_classes_} classes)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Classes', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
        # Add interpretation
        st.info("""
        **Coefficient Interpretation:**
        - **Red (Positive)**: Feature increases log-odds for this class
        - **Blue (Negative)**: Feature decreases log-odds for this class  
        - **White (Near Zero)**: Feature has little effect on this class
        - **Magnitude**: Larger absolute values indicate stronger effects
        
        **Multinomial Nature**: Each row shows how features affect one class 
        relative to the baseline (reference class in multinomial formulation).
        """)
    
    def plot_feature_importance(self, max_features=15, figsize=(10, 6)):
        """
        Plot aggregated feature importance across all classes
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features to display
        figsize : tuple
            Figure size for the plot
        """
        importance_data = self.get_feature_importance()
        if importance_data is None:
            st.warning("Feature importance not available")
            return
        
        # Get aggregated importance
        aggregated = importance_data['aggregated']
        feature_names = aggregated['sorted_features'][:max_features]
        importance_scores = aggregated['sorted_importance'][:max_features]
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importance_scores, color='skyblue', alpha=0.8)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score (Mean |Coefficient|)', fontsize=12)
        ax.set_title(f'Feature Importance - Multinomial Logistic Regression\n'
                    f'(Top {len(feature_names)} features)', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, importance_scores)):
            ax.text(bar.get_width() + max(importance_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontsize=10)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
        # Add interpretation
        st.info("""
        **Feature Importance Interpretation:**
        - Based on mean absolute coefficient across all classes
        - Higher scores indicate features with stronger overall influence
        - Represents linear relationship strength in log-odds space
        - Important for understanding which features drive multinomial classification
        """)
    
    def plot_probability_distribution(self, X_sample=None, figsize=(12, 6)):
        """
        Plot distribution of predicted probabilities
        
        Parameters:
        -----------
        X_sample : array-like, optional
            Sample data to analyze. If None, uses a subset of training data.
        figsize : tuple
            Figure size for the plot
        """
        if not self.is_fitted_:
            st.warning("Model must be fitted to plot probability distribution")
            return
        
        # Use provided sample or create one from training data
        if X_sample is None:
            st.warning("Probability distribution analysis requires sample data")
            return
        
        # Get probabilities
        probabilities = self.predict_proba(X_sample)
        max_probabilities = np.max(probabilities, axis=1)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Distribution of maximum probabilities
        ax1.hist(max_probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(max_probabilities), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probabilities):.3f}')
        ax1.set_xlabel('Maximum Probability', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Prediction Confidence', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Box plot of probabilities by predicted class
        prob_by_class = []
        class_labels = []
        for i, class_name in enumerate(self.classes_):
            class_mask = predicted_classes == i
            if np.sum(class_mask) > 0:
                prob_by_class.append(max_probabilities[class_mask])
                class_labels.append(str(class_name))
        
        if prob_by_class:
            ax2.boxplot(prob_by_class, labels=class_labels)
            ax2.set_xlabel('Predicted Class', fontsize=12)
            ax2.set_ylabel('Maximum Probability', fontsize=12)
            ax2.set_title('Prediction Confidence by Class', fontsize=13, fontweight='bold')
            ax2.grid(alpha=0.3)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Add statistics
        st.info(f"""
        **Probability Statistics:**
        - Mean Confidence: {np.mean(max_probabilities):.3f}
        - Std Confidence: {np.std(max_probabilities):.3f}
        - High Confidence (>0.8): {np.sum(max_probabilities > 0.8)/len(max_probabilities)*100:.1f}%
        - Low Confidence (<0.6): {np.sum(max_probabilities < 0.6)/len(max_probabilities)*100:.1f}%
        """)
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for organized configuration
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Core Parameters", 
            "⚙️ Advanced Options", 
            "📊 Multinomial Theory",
            "🔍 Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Multinomial Logistic Regression Configuration**")
            
            # Regularization parameters
            st.markdown("**🎛️ Regularization Settings**")
            
            penalty = st.selectbox(
                "Penalty Type:",
                options=['l2', 'l1', 'elasticnet', 'none'],
                index=['l2', 'l1', 'elasticnet', 'none'].index(self.penalty),
                help="Regularization norm: L2 (Ridge), L1 (Lasso), ElasticNet (L1+L2), or None",
                key=f"{key_prefix}_penalty"
            )
            
            C = st.number_input(
                "Regularization Strength (C):",
                value=float(self.C),
                min_value=1e-6,
                max_value=1e6,
                step=0.1,
                format="%.4f",
                help="Inverse regularization strength. Lower values = more regularization",
                key=f"{key_prefix}_C"
            )
            
            # Elastic Net ratio (only if elasticnet selected)
            if penalty == 'elasticnet':
                l1_ratio = st.slider(
                    "L1 Ratio (ElasticNet):",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(self.l1_ratio) if self.l1_ratio is not None else 0.5,
                    step=0.1,
                    help="0 = L2 only, 1 = L1 only, 0.5 = equal mix",
                    key=f"{key_prefix}_l1_ratio"
                )
            else:
                l1_ratio = None
            
            # Solver selection
            st.markdown("**🔧 Solver Configuration**")
            
            # Recommend solver based on penalty
            if penalty == 'elasticnet':
                available_solvers = ['saga']
                recommended = 'saga'
            elif penalty == 'l1':
                available_solvers = ['liblinear', 'saga']
                recommended = 'saga'
            elif penalty == 'l2':
                available_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
                recommended = 'lbfgs'
            else:  # none
                available_solvers = ['lbfgs', 'newton-cg', 'sag', 'saga']
                recommended = 'lbfgs'
            
            solver = st.selectbox(
                "Solver Algorithm:",
                options=available_solvers,
                index=available_solvers.index(recommended),
                help=f"Recommended: {recommended} for {penalty} penalty",
                key=f"{key_prefix}_solver"
            )
            
            # Convergence parameters
            st.markdown("**🎯 Convergence Settings**")
            
            max_iter = st.number_input(
                "Maximum Iterations:",
                value=int(self.max_iter),
                min_value=100,
                max_value=10000,
                step=100,
                help="Maximum number of iterations for solver convergence",
                key=f"{key_prefix}_max_iter"
            )
            
            tol = st.number_input(
                "Tolerance:",
                value=float(self.tol),
                min_value=1e-8,
                max_value=1e-2,
                step=1e-6,
                format="%.2e",
                help="Tolerance for stopping criteria",
                key=f"{key_prefix}_tol"
            )
            
            # Show penalty-solver compatibility
            st.info(f"""
            **Current Configuration:**
            - Penalty: {penalty}
            - Solver: {solver}
            - Multinomial: Always enabled (native multi-class)
            - Regularization: {1/C:.4f} (1/C)
            """)
        
        with tab2:
            st.markdown("**Advanced Multinomial Options**")
            
            # Feature scaling
            auto_scale_features = st.checkbox(
                "Automatic Feature Scaling",
                value=self.auto_scale_features,
                help="Apply StandardScaler to features (recommended for multinomial)",
                key=f"{key_prefix}_auto_scale_features"
            )
            
            # Class weighting
            class_weight_option = st.selectbox(
                "Class Weighting:",
                options=['none', 'balanced'],
                index=0 if self.class_weight is None else 1,
                help="Handle class imbalance: balanced uses inverse class frequencies",
                key=f"{key_prefix}_class_weight_option"
            )
            
            class_weight = None if class_weight_option == 'none' else 'balanced'
            
            # Intercept options
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=self.fit_intercept,
                help="Whether to calculate intercept term",
                key=f"{key_prefix}_fit_intercept"
            )
            
            if fit_intercept:
                intercept_scaling = st.number_input(
                    "Intercept Scaling:",
                    value=float(self.intercept_scaling),
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    help="Scaling factor for intercept regularization",
                    key=f"{key_prefix}_intercept_scaling"
                )
            else:
                intercept_scaling = 1.0
            
            # Performance options
            st.markdown("**⚡ Performance Options**")
            
            n_jobs = st.selectbox(
                "Parallel Jobs:",
                options=[None, 1, 2, 4, -1],
                index=0,
                help="-1 uses all available cores",
                key=f"{key_prefix}_n_jobs"
            )
            
            warm_start = st.checkbox(
                "Warm Start",
                value=self.warm_start,
                help="Reuse previous solution as initialization",
                key=f"{key_prefix}_warm_start"
            )
            
            verbose = st.selectbox(
                "Verbosity Level:",
                options=[0, 1],
                index=self.verbose,
                help="Control training output verbosity",
                key=f"{key_prefix}_verbose"
            )
            
            random_state = st.number_input(
                "Random Seed:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
        
        with tab3:
            st.markdown("**Multinomial Logistic Regression Theory**")
            
            st.info("""
            **Multinomial Distribution & Softmax:**
            • Native multi-class approach (not binary decomposition)
            • Uses softmax function: P(y=k|x) = exp(x·θₖ) / Σⱼ exp(x·θⱼ)
            • Estimates class probabilities that sum to 1
            • Single unified model for all classes
            
            **Key Advantages:**
            • True probabilistic interpretation
            • Coherent multi-class decision boundaries
            • No artificial class imbalance issues
            • Efficient training and prediction
            """)
            
            # Mathematical foundation
            if st.button("📐 Mathematical Foundation", key=f"{key_prefix}_math_foundation"):
                st.markdown("""
                **Multinomial Logistic Regression Mathematics:**
                
                **Probability Model:**
                P(y = k | x, θ) = exp(θₖᵀx + βₖ) / Σⱼ₌₁ᴷ exp(θⱼᵀx + βⱼ)
                
                **Log-Likelihood:**
                ℓ(θ) = Σᵢ₌₁ⁿ Σₖ₌₁ᴷ yᵢₖ log P(y = k | xᵢ, θ)
                
                **Gradient (for class k):**
                ∇θₖ ℓ = Σᵢ₌₁ⁿ (yᵢₖ - P(y = k | xᵢ, θ)) xᵢ
                
                **Regularization:**
                • L2: R(θ) = λ Σₖ Σⱼ θₖⱼ²
                • L1: R(θ) = λ Σₖ Σⱼ |θₖⱼ|
                • ElasticNet: R(θ) = λ₁ Σₖ Σⱼ |θₖⱼ| + λ₂ Σₖ Σⱼ θₖⱼ²
                """)
            
            # Comparison with binary approaches
            if st.button("🔄 vs Binary Decomposition", key=f"{key_prefix}_vs_binary"):
                st.markdown("""
                **Multinomial vs Binary Decomposition:**
                
                **One-vs-Rest Issues:**
                • Artificial class imbalance in binary problems
                • Inconsistent probability estimates
                • Decision boundaries may not be optimal
                
                **One-vs-One Issues:**
                • Quadratic number of classifiers
                • Complex voting aggregation
                • Higher computational cost
                
                **Multinomial Advantages:**
                • Single coherent model
                • Natural probability estimates
                • Optimal decision boundaries
                • Linear computational complexity
                • No class imbalance artifacts
                """)
            
            # Practical considerations
            if st.button("💡 Practical Guidelines", key=f"{key_prefix}_guidelines"):
                st.markdown("""
                **When to Use Multinomial Logistic Regression:**
                
                **Ideal Scenarios:**
                • 3-20 classes (sweet spot for multinomial)
                • Need probability estimates
                • Linear/near-linear class boundaries
                • Interpretable model required
                • Fast prediction needed
                
                **Feature Requirements:**
                • Numerical features (continuous preferred)
                • No extreme outliers
                • Features should be somewhat normalized
                • Linear relationships with log-odds
                
                **Performance Tips:**
                • Use feature scaling for best results
                • L2 regularization for stability
                • L1 for feature selection
                • Balance regularization strength with model complexity
                """)
        
        with tab4:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Multinomial Logistic Regression** - Native Multi-class Linear Model:
            • 🎯 True multinomial probability distribution
            • 🔄 Single unified model (not binary decomposition)
            • 📊 Softmax activation for class probabilities
            • ⚡ Efficient O(K×N×D) training complexity
            • 🎪 Excellent baseline for multi-class problems
            • 🔍 Highly interpretable coefficients
            
            **Technical Specifications:**
            • Loss Function: Multinomial Cross-entropy
            • Optimization: Gradient-based (LBFGS, SAG, SAGA)
            • Regularization: L1, L2, ElasticNet, None
            • Output: Class probabilities via softmax
            """)
            
            # Implementation details
            if st.button("🔧 Implementation Details", key=f"{key_prefix}_implementation"):
                st.markdown("""
                **Multinomial Implementation:**
                
                **Model Architecture:**
                • K coefficient vectors (one per class)
                • Shared feature space across all classes
                • Softmax normalization for probabilities
                • Optional intercept terms per class
                
                **Training Process:**
                1. Initialize coefficient matrix θ ∈ ℝᴷˣᴰ
                2. Compute softmax probabilities for all classes
                3. Calculate multinomial cross-entropy loss
                4. Add regularization penalty
                5. Update coefficients via gradient descent
                6. Repeat until convergence
                
                **Prediction Process:**
                1. Compute linear combination: zₖ = θₖᵀx + βₖ
                2. Apply softmax: P(y=k|x) = exp(zₖ) / Σⱼ exp(zⱼ)
                3. Return class with highest probability
                """)
            
            # Solver comparison
            if st.button("🚀 Solver Comparison", key=f"{key_prefix}_solvers"):
                st.markdown("""
                **Solver Algorithm Comparison:**
                
                **LBFGS (Limited-memory BFGS):**
                • Best for: Small-medium datasets, L2/no penalty
                • Convergence: Fast, stable
                • Memory: Low memory usage
                
                **SAG (Stochastic Average Gradient):**
                • Best for: Large datasets, L2 penalty only
                • Convergence: Fast for large data
                • Memory: Moderate memory usage
                
                **SAGA (SAG with improvements):**
                • Best for: Large datasets, any penalty
                • Convergence: Fastest for large sparse data
                • Memory: Higher memory usage
                
                **Newton-CG:**
                • Best for: Small datasets, high precision
                • Convergence: Very accurate
                • Memory: Low memory usage
                
                **Liblinear:**
                • Best for: High-dimensional sparse data
                • Convergence: Good for specific cases
                • Memory: Very efficient
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_algo_details"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "C": C,
            "penalty": penalty,
            "l1_ratio": l1_ratio,
            "solver": solver,
            "max_iter": max_iter,
            "tol": tol,
            "auto_scale_features": auto_scale_features,
            "class_weight": class_weight,
            "fit_intercept": fit_intercept,
            "intercept_scaling": intercept_scaling,
            "warm_start": warm_start,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return MultinomialLogisticRegressionPlugin(
            C=hyperparameters.get("C", self.C),
            penalty=hyperparameters.get("penalty", self.penalty),
            l1_ratio=hyperparameters.get("l1_ratio", self.l1_ratio),
            solver=hyperparameters.get("solver", self.solver),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            tol=hyperparameters.get("tol", self.tol),
            auto_scale_features=hyperparameters.get("auto_scale_features", self.auto_scale_features),
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            intercept_scaling=hyperparameters.get("intercept_scaling", self.intercept_scaling),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            verbose=hyperparameters.get("verbose", self.verbose),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Multinomial Logistic Regression"""
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
        """Check if Multinomial Logistic Regression is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Multinomial Logistic Regression requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            n_classes = len(unique_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). Consider reducing number of classes."
            
            # Check class distribution
            min_class_samples = min(np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes))
            if min_class_samples < 2:
                return False, "Each class needs at least 2 samples"
            
            # Multinomial specific advantages
            advantages = []
            considerations = []
            
            # Optimal class range for multinomial
            if n_classes == 2:
                advantages.append("Binary problem - multinomial reduces to standard logistic regression")
            elif 3 <= n_classes <= 10:
                advantages.append(f"Ideal for multinomial ({n_classes} classes) - natural multi-class approach")
            elif 11 <= n_classes <= 50:
                advantages.append(f"Good for multinomial ({n_classes} classes) - more efficient than binary decomposition")
            else:
                considerations.append(f"Many classes ({n_classes}) - ensure sufficient samples per class")
            
            # Sample size considerations
            samples_per_class = X.shape[0] / n_classes
            if samples_per_class >= 50:
                advantages.append("Excellent samples per class ratio")
            elif samples_per_class >= 20:
                advantages.append("Good samples per class ratio")
            else:
                considerations.append("Few samples per class - may need regularization")
            
            # Feature space considerations
            if X.shape[1] <= 100:
                advantages.append("Moderate feature space - good for multinomial interpretation")
            elif X.shape[1] <= 1000:
                advantages.append("High-dimensional data - multinomial handles well with regularization")
            else:
                considerations.append("Very high-dimensional - ensure proper regularization")
            
            # Class balance
            class_counts = np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes)
            imbalance_ratio = max(class_counts) / min(class_counts)
            if imbalance_ratio <= 3:
                advantages.append("Well-balanced classes - ideal for multinomial")
            elif imbalance_ratio <= 10:
                advantages.append("Moderate imbalance - manageable with class weights")
            else:
                considerations.append("High class imbalance - consider balanced class weights")
            
            # Build compatibility message
            efficiency_rating = ("Excellent" if 3 <= n_classes <= 10 else "Good" if n_classes <= 50 else "Moderate")
            
            message_parts = [
                f"✅ Compatible with {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes",
                f"🎯 Multinomial efficiency: {efficiency_rating}"
            ]
            
            if advantages:
                message_parts.append("🌟 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        params = {
            "n_features": self.n_features_in_,
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist(),
            "feature_names": self.feature_names_,
            "model_type": "Multinomial Logistic Regression",
            "native_multiclass": True,
            "probability_model": "Softmax (Multinomial)",
            "feature_scaling": self.scaler_ is not None,
            "regularization": {
                "penalty": self.penalty,
                "C": self.C,
                "l1_ratio": self.l1_ratio,
                "regularization_strength": 1.0 / self.C
            },
            "solver_config": {
                "solver": self.model_.solver,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "convergence": hasattr(self.model_, 'n_iter_')
            }
        }
        
        # Add convergence info if available
        if hasattr(self.model_, 'n_iter_'):
            iterations = self.model_.n_iter_[0] if hasattr(self.model_.n_iter_, '__len__') else self.model_.n_iter_
            params["solver_config"]["iterations_used"] = int(iterations)
            params["solver_config"]["converged"] = iterations < self.max_iter
        
        return params
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Multinomial Logistic Regression",
            "model_type": "Native Multi-class Linear Model",
            "training_completed": True,
            "multinomial_characteristics": {
                "native_multiclass": True,
                "softmax_probabilities": True,
                "unified_model": True,
                "linear_decision_boundaries": True,
                "probabilistic_output": True,
                "interpretable_coefficients": True
            },
            "model_configuration": {
                "penalty": self.penalty,
                "regularization_strength": 1.0 / self.C,
                "solver": self.model_.solver if self.is_fitted_ else self.solver,
                "feature_scaling": self.scaler_ is not None,
                "n_classes": self.n_classes_,
                "n_features": self.n_features_in_
            },
            "multinomial_analysis": self.get_multinomial_analysis(),
            "performance_info": {
                "training_accuracy": self.training_score_,
                "cv_scores": self.cv_scores_.tolist() if self.cv_scores_ is not None else None,
                "mean_cv_score": float(np.mean(self.cv_scores_)) if self.cv_scores_ is not None else None,
                "std_cv_score": float(np.std(self.cv_scores_)) if self.cv_scores_ is not None else None
            },
            "computational_characteristics": {
                "training_complexity": f"O({self.n_classes_} × N × D)",
                "prediction_complexity": f"O({self.n_classes_} × D)",
                "memory_usage": f"Stores {self.n_classes_} × {self.n_features_in_} coefficients",
                "scalability": "Linear in classes (vs quadratic for OvO)",
                "efficiency": "High - single unified model"
            },
            "multinomial_theory": {
                "probability_model": "P(y=k|x) = exp(θₖᵀx) / Σⱼ exp(θⱼᵀx)",
                "loss_function": "Multinomial Cross-entropy",
                "optimization": "Gradient-based (LBFGS/SAG/SAGA)",
                "decision_rule": "argmax over softmax probabilities",
                "advantages": "No class imbalance, coherent probabilities, efficient"
            }
        }
        
        return info
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get comprehensive algorithm information"""
        return {
            "name": self.get_name(),
            "category": self.get_category(),
            "description": self.get_description(),
            "type": "Multinomial Linear Classifier",
            "native_multiclass": True,
            "probability_support": True,
            "interpretability": "High",
            "computational_complexity": {
                "training": "O(K × N × D × I)",
                "prediction": "O(K × D)",
                "memory": "O(K × D)"
            },
            "hyperparameters": {
                "C": "Inverse regularization strength",
                "penalty": "Regularization type (l1, l2, elasticnet, none)",
                "solver": "Optimization algorithm",
                "max_iter": "Maximum iterations",
                "tol": "Convergence tolerance"
            },
            "strengths": [
                "True multinomial probability distribution",
                "No artificial class imbalance",
                "Highly interpretable coefficients",
                "Efficient single unified model",
                "Fast training and prediction",
                "Good baseline for multi-class problems"
            ],
            "limitations": [
                "Assumes linear decision boundaries",
                "Sensitive to outliers",
                "Requires feature scaling for optimal performance",
                "May struggle with highly non-linear patterns"
            ],
            "best_use_cases": [
                "Multi-class classification (3-50 classes)",
                "Need probability estimates",
                "Interpretability required",
                "Linear/near-linear class separation",
                "Baseline model for comparison",
                "Fast prediction requirements"
            ]
        }

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for Multinomial Logistic Regression.

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
            return {"error": "Model not fitted. Cannot retrieve Multinomial Logistic Regression specific metrics."}

        metrics = {}

        # From multinomial_analysis_
        if self.multinomial_analysis_:
            if 'convergence' in self.multinomial_analysis_ and isinstance(self.multinomial_analysis_['convergence'], dict):
                metrics['mlr_convergence_iterations'] = self.multinomial_analysis_['convergence'].get('iterations')

            if 'regularization_analysis' in self.multinomial_analysis_ and isinstance(self.multinomial_analysis_['regularization_analysis'], dict):
                metrics['mlr_coefficient_sparsity'] = self.multinomial_analysis_['regularization_analysis'].get('sparsity_level')
            
            if 'probability_analysis' in self.multinomial_analysis_ and isinstance(self.multinomial_analysis_['probability_analysis'], dict):
                metrics['mlr_mean_max_probability'] = self.multinomial_analysis_['probability_analysis'].get('mean_max_probability')

        # From cross-validation scores (if performed)
        if self.cv_scores_ is not None:
            try:
                metrics['mlr_internal_cv_mean_accuracy'] = float(np.mean(self.cv_scores_))
                metrics['mlr_internal_cv_std_accuracy'] = float(np.std(self.cv_scores_))
            except Exception: # Handle cases where cv_scores_ might not be numeric
                pass
        
        # Number of features used
        if self.n_features_in_ is not None:
            metrics['mlr_num_features_in'] = self.n_features_in_
        
        # Number of classes
        if self.n_classes_ is not None:
            metrics['mlr_num_classes'] = self.n_classes_

        # McFadden's Pseudo R-squared (requires y_true and y_proba)
        if y_true is not None and y_proba is not None:
            try:
                # Ensure y_true is numerically encoded (0, 1, ..., n_classes-1)
                # This plugin uses self.label_encoder_
                if self.label_encoder_:
                    y_true_encoded = self.label_encoder_.transform(y_true)
                else: # Should not happen if fit was called, but as a fallback
                    y_true_encoded = y_true 

                n_samples = len(y_true_encoded)
                
                # Log-likelihood of the full model
                # Ensure probabilities are clipped to avoid log(0)
                clipped_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
                
                # Select the probability of the true class for each sample
                log_likelihoods_model = np.log(clipped_proba[np.arange(n_samples), y_true_encoded])
                ll_model = np.sum(log_likelihoods_model)

                # Log-likelihood of the null model (intercept-only)
                # This predicts based on overall class proportions in y_true
                class_counts = np.bincount(y_true_encoded, minlength=self.n_classes_) # Ensure all classes are represented
                class_probas_null = class_counts / n_samples
                
                # Handle cases where a class might not be in y_true (though unlikely if y_true is from test set of original data)
                # We only sum log-likelihood for classes present in y_true_encoded (class_counts > 0)
                ll_null = 0
                for k_idx in range(self.n_classes_):
                    if class_counts[k_idx] > 0:
                         ll_null += class_counts[k_idx] * np.log(np.clip(class_probas_null[k_idx], 1e-15, 1))
                
                if ll_null == 0: 
                    # Avoid division by zero. If ll_null is 0, it means all samples belong to one class,
                    # or something is very wrong. If ll_model is also 0, it's a perfect fit by null.
                    metrics['mlr_mcfaddens_pseudo_r2'] = 1.0 if ll_model == 0 else 0.0
                elif ll_model > ll_null : # ll_model should not be greater than ll_null
                    metrics['mlr_mcfaddens_pseudo_r2'] = 0.0 
                else:
                    metrics['mlr_mcfaddens_pseudo_r2'] = float(1 - (ll_model / ll_null))
            except Exception as e:
                metrics['mlr_mcfaddens_pseudo_r2_error'] = str(e)

        if not metrics:
            metrics['info'] = "No specific Multinomial Logistic Regression metrics were available (e.g., analysis during fit failed or y_true/y_proba not provided for Pseudo R2)."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return MultinomialLogisticRegressionPlugin()