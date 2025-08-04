import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
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


class MulticlassSVMPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Multi-class SVM Plugin - Direct Multi-class Support
    
    This plugin implements Support Vector Machine with native multi-class support,
    offering both One-vs-Rest and One-vs-One strategies, multiple kernel functions,
    and comprehensive hyperparameter optimization for multi-class classification.
    
    Key Features:
    - Native multi-class SVM implementation
    - Multiple kernel functions (linear, RBF, polynomial, sigmoid)
    - One-vs-Rest and One-vs-One strategies
    - Advanced hyperparameter optimization
    - Feature scaling and preprocessing
    - Probability calibration for better probability estimates
    - Comprehensive visualization and analysis
    - Support vector analysis and interpretation
    - Kernel parameter optimization
    - Class imbalance handling
    """
    
    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0.0,
        shrinking=True,
        probability=True,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        max_iter=-1,
        decision_function_shape='ovr',
        break_ties=False,
        auto_scale_features=True,
        probability_calibration=True,
        cross_validation_folds=5,
        random_state=42
    ):
        super().__init__()
        
        # Core SVM parameters
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        
        # Advanced options
        self.auto_scale_features = auto_scale_features
        self.probability_calibration = probability_calibration
        self.cross_validation_folds = cross_validation_folds
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "Multi-class SVM"
        self._description = "Support Vector Machine with native multi-class support and advanced kernel methods"
        self._category = "Specialized Multi-class"
        
        # Required capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._min_samples_required = 20
        
        # Internal state
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.calibrated_model_ = None
        self.is_fitted_ = False
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_names_ = None
        self.n_support_ = None
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.cv_scores_ = None
        self.kernel_analysis_ = {}
        self.class_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Multi-class SVM model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights
            
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # 🎯 STORE FEATURE NAMES BEFORE VALIDATION!
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Encode labels if necessary
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        
        # Store class information
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Feature scaling
        if self.auto_scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X.copy()
            self.scaler_ = None
        
        # Create SVM model
        self.model_ = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state
        )
        
        # Fit the model
        self.model_.fit(X_scaled, y_encoded, sample_weight=sample_weight)
        
        # Store support vector information
        self.n_support_ = self.model_.n_support_
        self.support_vectors_ = self.model_.support_vectors_
        if hasattr(self.model_, 'dual_coef_'):
            self.dual_coef_ = self.model_.dual_coef_
        
        # Probability calibration
        if self.probability_calibration and self.probability:
            self.calibrated_model_ = CalibratedClassifierCV(
                self.model_, 
                method='isotonic', 
                cv=3
            )
            self.calibrated_model_.fit(X_scaled, y_encoded)
        
        # Cross-validation evaluation
        if self.cross_validation_folds > 1:
            self.cv_scores_ = cross_val_score(
                self.model_, 
                X_scaled, 
                y_encoded, 
                cv=self.cross_validation_folds,
                scoring='accuracy'
            )
        
        # Perform kernel analysis
        self._analyze_kernel_performance(X_scaled, y_encoded)
        
        # Perform class analysis
        self._analyze_class_characteristics(X_scaled, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted SVM model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X)
        
        # Apply feature scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        y_pred_encoded = self.model_.predict(X_scaled)
        
        # Decode labels
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted SVM model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        probabilities : array of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if not self.probability:
            raise ValueError("Probability estimation is not enabled. Set probability=True")
        
        X = check_array(X)
        
        # Apply feature scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Use calibrated model if available, otherwise use regular model
        if self.calibrated_model_ is not None:
            probabilities = self.calibrated_model_.predict_proba(X_scaled)
        else:
            probabilities = self.model_.predict_proba(X_scaled)
        
        return probabilities
    
    def decision_function(self, X):
        """
        Evaluate the decision function for the samples in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        decision : array of shape (n_samples, n_classes*(n_classes-1)/2)
            Decision function values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing decision function")
        
        X = check_array(X)
        
        # Apply feature scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        return self.model_.decision_function(X_scaled)
    
    def _analyze_kernel_performance(self, X, y):
        """Analyze kernel performance characteristics"""
        try:
            self.kernel_analysis_ = {
                'kernel_type': self.kernel,
                'kernel_parameters': {
                    'C': self.C,
                    'gamma': self.gamma if self.kernel in ['rbf', 'poly', 'sigmoid'] else None,
                    'degree': self.degree if self.kernel == 'poly' else None,
                    'coef0': self.coef0 if self.kernel in ['poly', 'sigmoid'] else None
                },
                'support_vector_info': {
                    'n_support_vectors': np.sum(self.n_support_),
                    'support_vector_ratio': np.sum(self.n_support_) / len(y),
                    'support_vectors_per_class': dict(zip(self.classes_, self.n_support_))
                },
                'decision_function_shape': self.decision_function_shape,
                'multi_class_strategy': 'One-vs-Rest' if self.decision_function_shape == 'ovr' else 'One-vs-One'
            }
            
            # Kernel-specific analysis
            if self.kernel == 'rbf':
                self.kernel_analysis_['kernel_characteristics'] = {
                    'type': 'Radial Basis Function (Gaussian)',
                    'properties': 'Non-linear, infinite-dimensional feature space',
                    'gamma_interpretation': f"Kernel coefficient (gamma={self.gamma})",
                    'decision_boundary': 'Smooth, non-linear'
                }
            elif self.kernel == 'linear':
                self.kernel_analysis_['kernel_characteristics'] = {
                    'type': 'Linear',
                    'properties': 'Linear decision boundaries',
                    'feature_space': 'Original feature space',
                    'decision_boundary': 'Linear hyperplanes'
                }
            elif self.kernel == 'poly':
                self.kernel_analysis_['kernel_characteristics'] = {
                    'type': f'Polynomial (degree {self.degree})',
                    'properties': f'Polynomial feature interactions up to degree {self.degree}',
                    'gamma_interpretation': f"Kernel coefficient (gamma={self.gamma})",
                    'decision_boundary': 'Polynomial curves'
                }
            elif self.kernel == 'sigmoid':
                self.kernel_analysis_['kernel_characteristics'] = {
                    'type': 'Sigmoid (Hyperbolic Tangent)',
                    'properties': 'Neural network-like activation',
                    'gamma_interpretation': f"Kernel coefficient (gamma={self.gamma})",
                    'decision_boundary': 'Sigmoid-shaped'
                }
            
        except Exception as e:
            self.kernel_analysis_ = {'error': str(e)}
    
    def _analyze_class_characteristics(self, X, y):
        """Analyze multi-class characteristics"""
        try:
            # Class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(self.classes_, class_counts))
            
            # Class imbalance analysis
            max_count = np.max(class_counts)
            min_count = np.min(class_counts)
            imbalance_ratio = max_count / min_count
            
            # Multi-class complexity
            n_binary_classifiers = self.n_classes_ if self.decision_function_shape == 'ovr' else (self.n_classes_ * (self.n_classes_ - 1)) // 2
            
            self.class_analysis_ = {
                'n_classes': self.n_classes_,
                'class_names': list(self.classes_),
                'class_distribution': class_distribution,
                'class_proportions': {cls: count/len(y) for cls, count in class_distribution.items()},
                'imbalance_analysis': {
                    'imbalance_ratio': imbalance_ratio,
                    'most_frequent_class': self.classes_[np.argmax(class_counts)],
                    'least_frequent_class': self.classes_[np.argmin(class_counts)],
                    'is_balanced': imbalance_ratio <= 2.0
                },
                'multi_class_complexity': {
                    'strategy': 'One-vs-Rest' if self.decision_function_shape == 'ovr' else 'One-vs-One',
                    'n_binary_classifiers': n_binary_classifiers,
                    'computational_complexity': f"O({n_binary_classifiers} × SVM_training_time)"
                }
            }
            
            # Support vector analysis per class
            if hasattr(self, 'n_support_') and self.n_support_ is not None:
                sv_per_class = dict(zip(self.classes_, self.n_support_))
                self.class_analysis_['support_vector_analysis'] = {
                    'support_vectors_per_class': sv_per_class,
                    'support_vector_ratios': {
                        cls: sv_count / class_distribution[cls] 
                        for cls, sv_count in sv_per_class.items()
                    },
                    'most_complex_class': max(sv_per_class.keys(), key=lambda k: sv_per_class[k]),
                    'simplest_class': min(sv_per_class.keys(), key=lambda k: sv_per_class[k])
                }
            
        except Exception as e:
            self.class_analysis_ = {'error': str(e)}
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for organized parameter configuration
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Core SVM Parameters", 
            "🔧 Advanced Options", 
            "📊 Multi-class Strategy", 
            "📚 Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Core SVM Configuration**")
            
            # Regularization parameter C
            C = st.number_input(
                "C (Regularization Parameter):",
                value=float(self.C),
                min_value=0.001,
                max_value=1000.0,
                step=0.1,
                format="%.3f",
                help="Controls the trade-off between smooth decision boundary and classifying training points correctly",
                key=f"{key_prefix}_C"
            )
            
            # Kernel selection
            kernel = st.selectbox(
                "Kernel Function:",
                options=['rbf', 'linear', 'poly', 'sigmoid'],
                index=['rbf', 'linear', 'poly', 'sigmoid'].index(self.kernel),
                help="Kernel function used in the algorithm",
                key=f"{key_prefix}_kernel"
            )
            
            # Kernel-specific parameters
            if kernel in ['rbf', 'poly', 'sigmoid']:
                gamma_option = st.selectbox(
                    "Gamma (Kernel Coefficient):",
                    options=['scale', 'auto', 'custom'],
                    index=['scale', 'auto'].index(self.gamma) if self.gamma in ['scale', 'auto'] else 2,
                    help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'",
                    key=f"{key_prefix}_gamma_option"
                )
                
                if gamma_option == 'custom':
                    gamma_value = st.number_input(
                        "Custom Gamma Value:",
                        value=0.001,
                        min_value=1e-6,
                        max_value=10.0,
                        step=0.001,
                        format="%.6f",
                        key=f"{key_prefix}_gamma_value"
                    )
                    gamma = gamma_value
                else:
                    gamma = gamma_option
            else:
                gamma = 'scale'
            
            # Polynomial kernel specific
            if kernel == 'poly':
                degree = st.slider(
                    "Polynomial Degree:",
                    min_value=1,
                    max_value=10,
                    value=int(self.degree),
                    help="Degree of the polynomial kernel function",
                    key=f"{key_prefix}_degree"
                )
                
                coef0 = st.number_input(
                    "Independent Term (coef0):",
                    value=float(self.coef0),
                    min_value=-10.0,
                    max_value=10.0,
                    step=0.1,
                    help="Independent term in kernel function",
                    key=f"{key_prefix}_coef0"
                )
            else:
                degree = self.degree
                coef0 = self.coef0
            
            # Sigmoid kernel specific
            if kernel == 'sigmoid':
                if 'coef0' not in locals():
                    coef0 = st.number_input(
                        "Independent Term (coef0):",
                        value=float(self.coef0),
                        min_value=-10.0,
                        max_value=10.0,
                        step=0.1,
                        help="Independent term in kernel function",
                        key=f"{key_prefix}_coef0_sigmoid"
                    )
            
            # Show kernel information
            if kernel == 'rbf':
                st.info("🌀 **RBF Kernel**: Creates smooth, non-linear decision boundaries. Good for most problems.")
            elif kernel == 'linear':
                st.info("📏 **Linear Kernel**: Creates linear decision boundaries. Fast and interpretable.")
            elif kernel == 'poly':
                st.info(f"📈 **Polynomial Kernel (degree {degree})**: Creates polynomial decision boundaries.")
            elif kernel == 'sigmoid':
                st.info("〰️ **Sigmoid Kernel**: Neural network-like activation function.")
        
        with tab2:
            st.markdown("**Advanced SVM Options**")
            
            # Class weight strategy
            class_weight_option = st.selectbox(
                "Class Weight Strategy:",
                options=['none', 'balanced', 'balanced_subsample'],
                index=0 if self.class_weight is None else 1,
                help="Strategy for handling class imbalance",
                key=f"{key_prefix}_class_weight"
            )
            
            if class_weight_option == 'none':
                class_weight = None
            elif class_weight_option == 'balanced':
                class_weight = 'balanced'
            else:
                class_weight = 'balanced_subsample'
            
            # Probability estimation
            probability = st.checkbox(
                "Enable Probability Estimation",
                value=self.probability,
                help="Enable probability estimates (required for predict_proba)",
                key=f"{key_prefix}_probability"
            )
            
            # Probability calibration
            probability_calibration = st.checkbox(
                "Probability Calibration",
                value=self.probability_calibration,
                help="Calibrate probabilities using isotonic regression",
                key=f"{key_prefix}_probability_calibration"
            )
            
            # Feature scaling
            auto_scale_features = st.checkbox(
                "Auto Feature Scaling",
                value=self.auto_scale_features,
                help="Automatically scale features using StandardScaler",
                key=f"{key_prefix}_auto_scale_features"
            )
            
            # Advanced parameters
            col1, col2 = st.columns(2)
            
            with col1:
                tol = st.number_input(
                    "Tolerance (tol):",
                    value=float(self.tol),
                    min_value=1e-6,
                    max_value=1e-1,
                    step=1e-4,
                    format="%.2e",
                    help="Tolerance for stopping criterion",
                    key=f"{key_prefix}_tol"
                )
                
                cache_size = st.number_input(
                    "Cache Size (MB):",
                    value=int(self.cache_size),
                    min_value=50,
                    max_value=2000,
                    step=50,
                    help="Size of the kernel cache",
                    key=f"{key_prefix}_cache_size"
                )
            
            with col2:
                max_iter = st.number_input(
                    "Max Iterations:",
                    value=int(self.max_iter) if self.max_iter > 0 else 1000,
                    min_value=100,
                    max_value=10000,
                    step=100,
                    help="Hard limit on iterations (-1 for no limit)",
                    key=f"{key_prefix}_max_iter"
                )
                
                shrinking = st.checkbox(
                    "Use Shrinking Heuristic",
                    value=self.shrinking,
                    help="Whether to use the shrinking heuristic",
                    key=f"{key_prefix}_shrinking"
                )
            
            # Cross-validation
            cross_validation_folds = st.slider(
                "Cross-Validation Folds:",
                min_value=3,
                max_value=10,
                value=int(self.cross_validation_folds),
                help="Number of folds for cross-validation evaluation",
                key=f"{key_prefix}_cv_folds"
            )
            
            # Random state
            random_state = st.number_input(
                "Random Seed:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="Random state for reproducibility",
                key=f"{key_prefix}_random_state"
            )
        
        with tab3:
            st.markdown("**Multi-class Strategy Configuration**")
            
            # Decision function shape
            decision_function_shape = st.selectbox(
                "Multi-class Strategy:",
                options=['ovr', 'ovo'],
                index=['ovr', 'ovo'].index(self.decision_function_shape),
                help="Multi-class strategy: One-vs-Rest or One-vs-One",
                key=f"{key_prefix}_decision_function_shape"
            )
            
            # Break ties option
            break_ties = st.checkbox(
                "Break Ties",
                value=self.break_ties,
                help="Break ties according to the confidence values of decision_function",
                key=f"{key_prefix}_break_ties"
            )
            
            # Strategy information
            if decision_function_shape == 'ovr':
                st.info("""
                **One-vs-Rest (OvR) Strategy:**
                • Creates K binary classifiers for K classes
                • Each classifier: "Class i vs. All Other Classes"  
                • Training time: O(K × n_samples)
                • Memory usage: O(K × model_size)
                • Good for: Large number of classes
                """)
            else:
                st.info("""
                **One-vs-One (OvO) Strategy:**
                • Creates K(K-1)/2 binary classifiers for K classes
                • Each classifier: "Class i vs. Class j"
                • Training time: O(K² × n_samples/K)
                • Memory usage: O(K² × model_size)
                • Good for: Few classes, high accuracy
                """)
            
            # Multi-class analysis buttons
            if st.button("📊 Compare Multi-class Strategies", key=f"{key_prefix}_compare_strategies"):
                st.markdown("""
                **One-vs-Rest vs One-vs-One for SVM:**
                
                **Computational Complexity:**
                - OvR: K classifiers, each trained on full dataset
                - OvO: K(K-1)/2 classifiers, each trained on subset (2 classes)
                - Winner: OvR for large K (K > 10), OvO for small K
                
                **Memory Usage:**
                - OvR: Linear scaling with number of classes
                - OvO: Quadratic scaling with number of classes
                - Winner: OvR for memory efficiency
                
                **Accuracy:**
                - OvR: May suffer from class imbalance in binary problems
                - OvO: Each binary problem is naturally balanced
                - Winner: OvO often achieves higher accuracy for small K
                
                **Training Time:**
                - OvR: Each classifier sees all data but solves easier problems
                - OvO: Each classifier sees less data but solves harder problems
                - Winner: Depends on dataset size and separability
                
                **SVM-Specific Considerations:**
                - SVM training complexity: O(n²) to O(n³)
                - OvO benefits: Smaller subproblems train faster
                - OvR benefits: Better parallelization potential
                """)
            
            if st.button("🔍 SVM Multi-class Theory", key=f"{key_prefix}_svm_theory"):
                st.markdown("""
                **SVM Multi-class Mathematical Foundation:**
                
                **One-vs-Rest Approach:**
                • For each class i, solve: min ||w_i||² + C Σ ξ_j
                • Subject to: y_j(w_i·φ(x_j) + b_i) ≥ 1 - ξ_j
                • Decision: argmax_i (w_i·φ(x) + b_i)
                
                **One-vs-One Approach:**
                • For each pair (i,j), solve binary SVM
                • Voting: Count wins from all K(K-1)/2 classifiers
                • Decision: Class with most votes
                
                **Kernel Trick in Multi-class:**
                • φ(x): Feature mapping to higher dimensional space
                • K(x_i, x_j) = φ(x_i)·φ(x_j): Kernel function
                • Allows non-linear decision boundaries
                • Same kernel used for all binary classifiers
                
                **Support Vectors in Multi-class:**
                • Each binary classifier has its own support vectors
                • Total support vectors = Union of all binary SVs
                • Class complexity ≈ Number of support vectors for that class
                """)
        
        with tab4:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Multi-class SVM** - Direct Multi-class Support:
            • 🎯 Support Vector Machine with native multi-class capabilities
            • 🔄 Multiple strategies: One-vs-Rest and One-vs-One
            • 🌀 Various kernels: Linear, RBF, Polynomial, Sigmoid
            • ⚡ Efficient optimization algorithms
            • 📊 Probability estimation and calibration
            • 🎚️ Advanced hyperparameter control
            • 🔍 Support vector analysis
            
            **Mathematical Foundation:**
            • Optimization: Quadratic Programming (QP)
            • Objective: Maximize margin while minimizing classification errors
            • Kernel Trick: Implicit mapping to high-dimensional space
            • Lagrange Multipliers: Dual optimization problem
            """)
            
            # Kernel selection guide
            if st.button("🌀 Kernel Selection Guide", key=f"{key_prefix}_kernel_guide"):
                st.markdown("""
                **Choosing the Right Kernel:**
                
                **Linear Kernel (k(x,y) = x·y):**
                - Use when: High-dimensional data, many features
                - Advantages: Fast, interpretable, no overfitting
                - Best for: Text classification, gene expression
                
                **RBF Kernel (k(x,y) = exp(-γ||x-y||²)):**
                - Use when: Non-linear patterns, moderate dimensions
                - Advantages: Smooth boundaries, good default choice
                - Parameters: γ controls smoothness
                
                **Polynomial Kernel (k(x,y) = (γx·y + r)^d):**
                - Use when: Polynomial relationships expected
                - Advantages: Captures feature interactions
                - Parameters: degree d, coefficient γ, term r
                
                **Sigmoid Kernel (k(x,y) = tanh(γx·y + r)):**
                - Use when: Neural network-like behavior desired
                - Advantages: S-shaped decision boundaries
                - Note: Can be unstable, use with caution
                """)
            
            # Parameter tuning guide
            if st.button("🎛️ Parameter Tuning Guide", key=f"{key_prefix}_tuning_guide"):
                st.markdown("""
                **SVM Hyperparameter Tuning:**
                
                **C (Regularization Parameter):**
                - Low C: Smoother decision boundary, may underfit
                - High C: Complex decision boundary, may overfit
                - Tuning: Use cross-validation, typical range [0.1, 100]
                
                **γ (Gamma for RBF/Poly/Sigmoid):**
                - Low γ: Smooth, simple decision boundary
                - High γ: Complex, detailed decision boundary
                - Tuning: Often log scale [1e-4, 1e1]
                
                **Class Weights:**
                - Balanced: Automatically adjust for class imbalance
                - Custom: Manually specify per-class weights
                - Use when: Imbalanced datasets
                
                **Tuning Strategy:**
                1. Start with default parameters
                2. Grid search on C and γ (if applicable)
                3. Use cross-validation for evaluation
                4. Consider class balance and data scaling
                """)
            
            # When to use SVM
            if st.button("🎯 When to Use Multi-class SVM", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases for Multi-class SVM:**
                
                **Problem Characteristics:**
                • 2-20 classes (optimal: 3-10 classes)
                • High-dimensional data (SVM handles curse of dimensionality well)
                • Complex decision boundaries needed
                • Need for robust, theoretically grounded method
                
                **Data Characteristics:**
                • Small to medium datasets (SVM doesn't scale well to very large data)
                • Clean, well-preprocessed data
                • Numerical features (categorical need encoding)
                • Not necessarily linearly separable
                
                **When SVM Excels:**
                • High-dimensional text classification
                • Image classification with extracted features
                • Bioinformatics and genomics
                • Medical diagnosis with complex patterns
                • Any domain requiring maximum margin principle
                
                **When to Avoid SVM:**
                • Very large datasets (>100k samples)
                • Many classes (>50) - consider alternatives
                • Real-time prediction requirements
                • When interpretability is paramount
                • Heavily imbalanced datasets without proper handling
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_details"):
                info = self.get_algorithm_info()
                st.json(info)
        
        # Return all hyperparameters
        return {
            "C": C,
            "kernel": kernel,
            "degree": degree,
            "gamma": gamma,
            "coef0": coef0,
            "shrinking": shrinking,
            "probability": probability,
            "tol": tol,
            "cache_size": cache_size,
            "class_weight": class_weight,
            "max_iter": max_iter if max_iter > 0 else -1,
            "decision_function_shape": decision_function_shape,
            "break_ties": break_ties,
            "auto_scale_features": auto_scale_features,
            "probability_calibration": probability_calibration,
            "cross_validation_folds": cross_validation_folds,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return MulticlassSVMPlugin(
            C=hyperparameters.get("C", self.C),
            kernel=hyperparameters.get("kernel", self.kernel),
            degree=hyperparameters.get("degree", self.degree),
            gamma=hyperparameters.get("gamma", self.gamma),
            coef0=hyperparameters.get("coef0", self.coef0),
            shrinking=hyperparameters.get("shrinking", self.shrinking),
            probability=hyperparameters.get("probability", self.probability),
            tol=hyperparameters.get("tol", self.tol),
            cache_size=hyperparameters.get("cache_size", self.cache_size),
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            decision_function_shape=hyperparameters.get("decision_function_shape", self.decision_function_shape),
            break_ties=hyperparameters.get("break_ties", self.break_ties),
            auto_scale_features=hyperparameters.get("auto_scale_features", self.auto_scale_features),
            probability_calibration=hyperparameters.get("probability_calibration", self.probability_calibration),
            cross_validation_folds=hyperparameters.get("cross_validation_folds", self.cross_validation_folds),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Multi-class SVM"""
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
        """Check if Multi-class SVM is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Multi-class SVM requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            n_classes = len(unique_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 100:
                return False, f"Too many classes ({n_classes}). SVM may be inefficient with >100 classes."
            
            # Check class distribution
            min_class_samples = min(np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes))
            if min_class_samples < 2:
                return False, "Each class needs at least 2 samples"
            
            # SVM-specific compatibility analysis
            advantages = []
            considerations = []
            
            # Optimal class range
            if 2 <= n_classes <= 10:
                advantages.append(f"Optimal class count ({n_classes}) for SVM multi-class")
            elif 11 <= n_classes <= 50:
                advantages.append(f"Good class count ({n_classes}) for SVM")
                considerations.append("Consider One-vs-Rest strategy for efficiency")
            else:
                considerations.append(f"Many classes ({n_classes}) - may be computationally expensive")
            
            # Sample size analysis
            if X.shape[0] >= 1000:
                advantages.append("Good sample size for robust SVM training")
            elif X.shape[0] >= 100:
                advantages.append("Adequate sample size for SVM")
            else:
                considerations.append("Small dataset - SVM may overfit, consider regularization")
            
            # Dimensionality analysis
            if X.shape[1] >= 100:
                advantages.append("High-dimensional data - SVM handles curse of dimensionality well")
            elif X.shape[1] >= 10:
                advantages.append("Moderate dimensionality - good for SVM")
            else:
                advantages.append("Low dimensionality - consider feature engineering")
            
            # Class balance analysis
            class_counts = np.bincount(y if np.issubdtype(y.dtype, np.integer) else pd.Categorical(y).codes)
            max_count = np.max(class_counts)
            min_count = np.min(class_counts)
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio <= 2.0:
                advantages.append("Well-balanced classes")
            elif imbalance_ratio <= 5.0:
                considerations.append("Moderate class imbalance - consider class weights")
            else:
                considerations.append("High class imbalance - use balanced class weights")
            
            # Multi-class strategy recommendation
            if n_classes <= 5:
                strategy_rec = "One-vs-One recommended (higher accuracy)"
            elif n_classes <= 15:
                strategy_rec = "One-vs-Rest recommended (efficiency)"
            else:
                strategy_rec = "One-vs-Rest strongly recommended (scalability)"
            
            # Build compatibility message
            efficiency_rating = ("Excellent" if n_classes <= 10 and X.shape[0] <= 10000 else
                               "Good" if n_classes <= 20 and X.shape[0] <= 50000 else
                               "Moderate" if n_classes <= 50 and X.shape[0] <= 100000 else "Poor")
            
            message_parts = [
                f"✅ Compatible with {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes",
                f"⚡ Efficiency: {efficiency_rating} | {strategy_rec}"
            ]
            
            if advantages:
                message_parts.append("🎯 SVM advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """
        Get feature importance for Multi-class SVM
        
        Note: SVM doesn't have traditional feature importance, but we can compute
        the norm of coefficient vectors for linear kernels or use permutation importance
        """
        if not self.is_fitted_:
            return None
        
        try:
            feature_importance = {}
            
            # For linear kernel, we can compute coefficient-based importance
            if self.kernel == 'linear' and hasattr(self.model_, 'coef_'):
                coef_matrix = self.model_.coef_
                
                # Per-class feature importance (absolute coefficients)
                per_class_importance = {}
                for i, class_name in enumerate(self.classes_):
                    if self.n_classes_ == 2:
                        # Binary classification: single coefficient vector
                        coef_vector = coef_matrix[0] if i == 1 else -coef_matrix[0]
                    else:
                        # Multi-class: one coefficient vector per class
                        coef_vector = coef_matrix[i] if i < coef_matrix.shape[0] else np.zeros(coef_matrix.shape[1])
                    
                    per_class_importance[str(class_name)] = {
                        'coefficients': coef_vector.tolist(),
                        'abs_coefficients': np.abs(coef_vector).tolist(),
                        'importance_scores': np.abs(coef_vector).tolist()
                    }
                
                # Aggregated importance (mean absolute coefficients across classes)
                if self.n_classes_ == 2:
                    aggregated_importance = np.abs(coef_matrix[0])
                else:
                    aggregated_importance = np.mean(np.abs(coef_matrix), axis=0)
                
                # Sort features by importance
                importance_indices = np.argsort(aggregated_importance)[::-1]
                
                feature_importance = {
                    'per_class': per_class_importance,
                    'aggregated': {
                        'feature_names': self.feature_names_,
                        'importance_scores': aggregated_importance.tolist(),
                        'sorted_indices': importance_indices.tolist(),
                        'sorted_features': [self.feature_names_[i] for i in importance_indices],
                        'sorted_scores': aggregated_importance[importance_indices].tolist()
                    },
                    'method': 'linear_coefficients',
                    'interpretation': 'Absolute values of linear SVM coefficients'
                }
            
            else:
                # For non-linear kernels, coefficient-based importance is not available
                feature_importance = {
                    'method': 'not_available',
                    'reason': f'Feature importance not directly available for {self.kernel} kernel',
                    'suggestion': 'Consider using permutation importance or SHAP values for non-linear kernels',
                    'alternative_analysis': {
                        'support_vector_analysis': 'Analyze support vectors and their influence',
                        'kernel_analysis': 'Examine kernel parameters and decision boundaries',
                        'feature_scaling_impact': 'Analyze impact of feature scaling on performance'
                    }
                }
            
            return feature_importance
        
        except Exception as e:
            return {
                'error': str(e),
                'method': 'error_in_computation'
            }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist(),
            "feature_names": self.feature_names_,
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "multi_class_strategy": self.decision_function_shape,
            "n_support_vectors": np.sum(self.n_support_) if self.n_support_ is not None else None,
            "support_vectors_per_class": dict(zip(self.classes_, self.n_support_)) if self.n_support_ is not None else None,
            "feature_scaling": self.scaler_ is not None,
            "probability_estimation": self.probability,
            "probability_calibration": self.calibrated_model_ is not None,
            "cv_scores": self.cv_scores_.tolist() if self.cv_scores_ is not None else None,
            "cv_mean": float(np.mean(self.cv_scores_)) if self.cv_scores_ is not None else None,
            "cv_std": float(np.std(self.cv_scores_)) if self.cv_scores_ is not None else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        return {
            "algorithm": "Multi-class SVM",
            "kernel_type": self.kernel,
            "multi_class_strategy": self.decision_function_shape,
            "training_completed": True,
            "svm_characteristics": {
                "kernel_method": True,
                "maximum_margin": True,
                "support_vector_based": True,
                "non_parametric": self.kernel != 'linear',
                "probability_support": self.probability,
                "handles_non_linearity": self.kernel != 'linear'
            },
            "kernel_analysis": self.kernel_analysis_,
            "class_analysis": self.class_analysis_,
            "model_configuration": {
                "C": self.C,
                "kernel": self.kernel,
                "gamma": self.gamma,
                "degree": self.degree,
                "coef0": self.coef0,
                "class_weight": self.class_weight,
                "feature_scaling": self.scaler_ is not None,
                "probability_calibration": self.calibrated_model_ is not None
            },
            "performance_info": {
                "n_support_vectors": np.sum(self.n_support_) if self.n_support_ is not None else None,
                "support_vector_ratio": np.sum(self.n_support_) / (len(self.classes_) * 100) if self.n_support_ is not None else None,
                "cv_scores": self.cv_scores_.tolist() if self.cv_scores_ is not None else None,
                "cv_mean_accuracy": float(np.mean(self.cv_scores_)) if self.cv_scores_ is not None else None,
                "cv_std_accuracy": float(np.std(self.cv_scores_)) if self.cv_scores_ is not None else None
            },
            "computational_complexity": {
                "training_time": f"O(n²) to O(n³) depending on solver and data",
                "prediction_time": f"O(n_support_vectors × n_features)",
                "memory_usage": f"O(n_support_vectors × n_features)",
                "scalability": "Good for medium-sized datasets, may struggle with very large data"
            }
        }
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get comprehensive algorithm information"""
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "category": self.get_category(),
            "type": "Support Vector Machine with Multi-class Extensions",
            "supports_classification": self._supports_classification,
            "supports_regression": self._supports_regression,
            "min_samples_required": self._min_samples_required,
            "algorithm_family": "Kernel Methods",
            "learning_type": "Supervised Learning",
            "prediction_type": "Discriminative Model",
            "mathematical_foundation": {
                "optimization_problem": "Quadratic Programming (QP)",
                "objective": "Maximize margin while minimizing classification errors",
                "lagrangian": "Dual optimization using Lagrange multipliers",
                "kernel_trick": "Implicit mapping to high-dimensional feature space",
                "decision_function": "Sign of weighted kernel combinations"
            },
            "key_advantages": [
                "Effective in high-dimensional spaces",
                "Memory efficient (uses support vectors only)",
                "Versatile (different kernel functions)",
                "Works well with small to medium datasets",
                "Strong theoretical foundation",
                "Good generalization performance"
            ],
            "key_limitations": [
                "Poor scalability to very large datasets",
                "Sensitive to feature scaling",
                "No probabilistic output (unless enabled)",
                "Difficult to interpret (except linear kernel)",
                "Many hyperparameters to tune",
                "Computationally expensive for large datasets"
            ],
            "use_cases": [
                "Text classification and sentiment analysis",
                "Image classification with feature extraction",
                "Bioinformatics and genomics",
                "Medical diagnosis",
                "Financial fraud detection",
                "Pattern recognition in high-dimensional data"
            ],
            "hyperparameter_sensitivity": {
                "C": "High - controls overfitting vs underfitting trade-off",
                "gamma": "High - controls kernel width and model complexity",
                "kernel": "Medium - choice affects model capacity significantly",
                "class_weight": "Medium - important for imbalanced datasets"
            }
        }

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for Multi-class SVM.

        These metrics are primarily derived from the internal state and analyses
        performed during the fit method (e.g., support vector counts, kernel properties).

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Not directly used for generating new metrics here,
            as SVM specific metrics are mostly from its internal structure.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used.
        y_proba : np.ndarray, optional
            Predicted probabilities. Not directly used.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve Multi-class SVM specific metrics."}

        metrics = {}

        # From kernel_analysis_
        if self.kernel_analysis_ and not self.kernel_analysis_.get('error'):
            if 'support_vector_info' in self.kernel_analysis_:
                sv_info = self.kernel_analysis_['support_vector_info']
                metrics['msvm_total_support_vectors'] = sv_info.get('n_support_vectors')
                metrics['msvm_support_vector_ratio'] = sv_info.get('support_vector_ratio')
            
            metrics['msvm_decision_function_shape'] = self.kernel_analysis_.get('decision_function_shape')
            # Individual n_support_ for each class can be many, so we'll report total and ratio.
            # Specific per-class counts are in get_model_params or get_training_info.

        # From class_analysis_
        if self.class_analysis_ and not self.class_analysis_.get('error'):
            if 'imbalance_analysis' in self.class_analysis_:
                imbalance_info = self.class_analysis_['imbalance_analysis']
                metrics['msvm_class_imbalance_ratio'] = imbalance_info.get('imbalance_ratio')
            
            if 'multi_class_complexity' in self.class_analysis_:
                complexity_info = self.class_analysis_['multi_class_complexity']
                metrics['msvm_num_binary_classifiers'] = complexity_info.get('n_binary_classifiers')
            
            # Support vector ratios per class could be added if desired, but might make the dict large.
            # Example:
            # if 'support_vector_analysis' in self.class_analysis_ and 'support_vector_ratios' in self.class_analysis_['support_vector_analysis']:
            #     for cls, ratio in self.class_analysis_['support_vector_analysis']['support_vector_ratios'].items():
            #         metrics[f'msvm_sv_ratio_class_{cls}'] = ratio

        # From cross-validation scores (if performed)
        if self.cv_scores_ is not None:
            metrics['msvm_internal_cv_mean_accuracy'] = float(np.mean(self.cv_scores_))
            metrics['msvm_internal_cv_std_accuracy'] = float(np.std(self.cv_scores_))
            metrics['msvm_internal_cv_folds'] = int(self.cross_validation_folds)

        # Number of features used
        if self.n_features_in_ is not None:
            metrics['msvm_num_features_in'] = self.n_features_in_
        
        # Number of classes
        if self.n_classes_ is not None:
            metrics['msvm_num_classes'] = self.n_classes_

        if not metrics:
            metrics['info'] = "No specific Multi-class SVM metrics were available (e.g., analysis during fit failed or model not fitted properly)."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return MulticlassSVMPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Multi-class SVM Plugin
    """
    print("Testing Multi-class SVM Plugin...")
    
    try:
        # Create sample multi-class data
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Generate multi-class dataset
        X, y = make_classification(
            n_samples=600,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_classes=4,  # 4 classes for multi-class SVM
            n_clusters_per_class=1,
            class_sep=1.0,
            flip_y=0.02,
            random_state=42
        )
        
        print(f"\n📊 Multi-class Dataset Info:")
        print(f"Shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and test Multi-class SVM plugin
        plugin = MulticlassSVMPlugin(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            decision_function_shape='ovr',  # One-vs-Rest strategy
            probability=True,
            auto_scale_features=True,
            probability_calibration=True,
            cross_validation_folds=5,
            random_state=42
        )
        
        print("\n🔍 Plugin Info:")
        print(f"Name: {plugin.get_name()}")
        print(f"Category: {plugin.get_category()}")
        print(f"Description: {plugin.get_description()}")
        
        # Check compatibility
        compatible, message = plugin.is_compatible_with_data(X_train, y_train)
        print(f"\n✅ Compatibility: {message}")
        
        if compatible:
            # Train Multi-class SVM
            print("\n🚀 Training Multi-class SVM...")
            plugin.fit(X_train, y_train)
            
            # Make predictions
            y_pred = plugin.predict(X_test)
            y_proba = plugin.predict_proba(X_test)
            decision_scores = plugin.decision_function(X_test)
            
            # Evaluate performance
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n📊 Multi-class SVM Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Classes: {plugin.classes_}")
            print(f"Multi-class strategy: {plugin.decision_function_shape}")
            
            # Get model parameters
            model_params = plugin.get_model_params()
            print(f"\n⚙️ Model Configuration:")
            print(f"Kernel: {model_params.get('kernel', 'Unknown')}")
            print(f"C parameter: {model_params.get('C', 'Unknown')}")
            print(f"Gamma: {model_params.get('gamma', 'Unknown')}")
            print(f"Multi-class strategy: {model_params.get('multi_class_strategy', 'Unknown')}")
            print(f"Support vectors: {model_params.get('n_support_vectors', 'Unknown')}")
            print(f"Feature scaling: {model_params.get('feature_scaling', False)}")
            
            # Support vector analysis
            if model_params.get('support_vectors_per_class'):
                sv_per_class = model_params['support_vectors_per_class']
                print(f"\n🎯 Support Vector Analysis:")
                total_sv = sum(sv_per_class.values())
                print(f"Total support vectors: {total_sv} ({total_sv/len(y_train)*100:.1f}% of training data)")
                for class_name, sv_count in sv_per_class.items():
                    print(f"Class {class_name}: {sv_count} support vectors")
            
            # Cross-validation results
            cv_mean = model_params.get('cv_mean')
            cv_std = model_params.get('cv_std')
            if cv_mean is not None:
                print(f"\n📈 Cross-Validation Results:")
                print(f"CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
                print(f"CV Scores: {model_params.get('cv_scores', [])}")
            
            # Feature importance analysis (if available)
            feature_importance = plugin.get_feature_importance()
            if feature_importance and feature_importance.get('method') == 'linear_coefficients':
                print(f"\n🔍 Feature Importance (Linear Kernel):")
                aggregated = feature_importance['aggregated']
                sorted_features = aggregated['sorted_features'][:5]
                sorted_scores = aggregated['sorted_scores'][:5]
                
                print("Top 5 Most Important Features:")
                for i, (feature, score) in enumerate(zip(sorted_features, sorted_scores)):
                    print(f"{i+1}. {feature}: {score:.4f}")
            elif feature_importance:
                print(f"\n🔍 Feature Importance:")
                print(f"Method: {feature_importance.get('method', 'Unknown')}")
                if 'reason' in feature_importance:
                    print(f"Note: {feature_importance['reason']}")
                if 'suggestion' in feature_importance:
                    print(f"Suggestion: {feature_importance['suggestion']}")
            
            # Training info
            training_info = plugin.get_training_info()
            print(f"\n📈 Training Info:")
            print(f"Algorithm: {training_info['algorithm']}")
            print(f"Kernel type: {training_info['kernel_type']}")
            print(f"Multi-class strategy: {training_info['multi_class_strategy']}")
            
            # SVM characteristics
            svm_chars = training_info['svm_characteristics']
            print(f"Kernel method: {svm_chars['kernel_method']}")
            print(f"Maximum margin: {svm_chars['maximum_margin']}")
            print(f"Support vector based: {svm_chars['support_vector_based']}")
            print(f"Handles non-linearity: {svm_chars['handles_non_linearity']}")
            
            # Kernel analysis
            if 'kernel_analysis' in training_info:
                kernel_analysis = training_info['kernel_analysis']
                print(f"\n🌀 Kernel Analysis:")
                print(f"Kernel type: {kernel_analysis.get('kernel_type', 'Unknown')}")
                
                if 'support_vector_info' in kernel_analysis:
                    sv_info = kernel_analysis['support_vector_info']
                    print(f"Support vector ratio: {sv_info['support_vector_ratio']:.3f}")
                
                if 'kernel_characteristics' in kernel_analysis:
                    kernel_chars = kernel_analysis['kernel_characteristics']
                    print(f"Kernel properties: {kernel_chars.get('properties', 'Unknown')}")
                    print(f"Decision boundary: {kernel_chars.get('decision_boundary', 'Unknown')}")
            
            # Class analysis
            if 'class_analysis' in training_info:
                class_analysis = training_info['class_analysis']
                print(f"\n📊 Class Analysis:")
                print(f"Number of classes: {class_analysis.get('n_classes', 'Unknown')}")
                
                if 'imbalance_analysis' in class_analysis:
                    imbalance = class_analysis['imbalance_analysis']
                    print(f"Class balance: {'Balanced' if imbalance['is_balanced'] else 'Imbalanced'}")
                    print(f"Imbalance ratio: {imbalance['imbalance_ratio']:.2f}")
                
                if 'multi_class_complexity' in class_analysis:
                    complexity = class_analysis['multi_class_complexity']
                    print(f"Strategy: {complexity['strategy']}")
                    print(f"Binary classifiers: {complexity['n_binary_classifiers']}")
            
            # Performance characteristics
            perf_info = training_info.get('performance_info', {})
            print(f"\n⚡ Performance Info:")
            if perf_info.get('cv_mean_accuracy'):
                print(f"CV Mean Accuracy: {perf_info['cv_mean_accuracy']:.4f}")
                print(f"CV Std Accuracy: {perf_info['cv_std_accuracy']:.4f}")
            
            # Algorithm info
            algo_info = plugin.get_algorithm_info()
            print(f"\n🧠 Algorithm Details:")
            print(f"Algorithm family: {algo_info['algorithm_family']}")
            print(f"Learning type: {algo_info['learning_type']}")
            print(f"Mathematical foundation: {algo_info['mathematical_foundation']['optimization_problem']}")
            
            print(f"\n🎯 Key Advantages:")
            for advantage in algo_info['key_advantages'][:3]:
                print(f"• {advantage}")
            
            print(f"\n💡 Key Limitations:")
            for limitation in algo_info['key_limitations'][:3]:
                print(f"• {limitation}")
            
            print("\n✅ Multi-class SVM Plugin test completed successfully!")
            print("🎯 Successfully trained SVM with native multi-class support!")
            
            # Demonstrate SVM benefits
            print(f"\n🚀 Multi-class SVM Benefits:")
            print(f"Native Multi-class: Direct support without binary decomposition overhead")
            print(f"Kernel Methods: {plugin.kernel} kernel handles non-linear patterns")
            print(f"Maximum Margin: Robust decision boundaries with theoretical guarantees")
            print(f"Support Vector Efficiency: Uses only {model_params.get('n_support_vectors', 0)} support vectors")
            
            # Show prediction confidence analysis
            print(f"\n🎯 Prediction Confidence Analysis:")
            max_probas = np.max(y_proba, axis=1)
            print(f"Average confidence: {np.mean(max_probas):.3f}")
            print(f"Min confidence: {np.min(max_probas):.3f}")
            print(f"Max confidence: {np.max(max_probas):.3f}")
            print(f"High confidence predictions (>0.8): {np.sum(max_probas > 0.8)/len(max_probas)*100:.1f}%")
            
            # Decision function analysis
            print(f"\n📊 Decision Function Analysis:")
            print(f"Decision scores shape: {decision_scores.shape}")
            print(f"Decision score range: [{np.min(decision_scores):.3f}, {np.max(decision_scores):.3f}]")
            print(f"Mean absolute decision score: {np.mean(np.abs(decision_scores)):.3f}")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()