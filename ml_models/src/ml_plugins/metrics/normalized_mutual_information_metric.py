import streamlit as st
import numpy as np
from typing import Any, Optional
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

# Import for plugin system - will be auto-fixed during save
try:
    from src.ml_plugins.base_metric_plugin import MetricPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_metric_plugin import MetricPlugin

class NormalizedMutualInformationMetric(MetricPlugin):
    def __init__(self):
        super().__init__()
        self._name = "Normalized Mutual Information"
        self._description = "Measures the mutual information between true and predicted labels, normalized by their entropies"
        self._category = "Information Theory"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = False
        self._higher_is_better = True
        self._range = (0.0, 1.0)
        
    def get_name(self) -> str:
        return self._name
        
    def get_description(self) -> str:
        return self._description
        
    def get_category(self) -> str:
        return self._category
        
    def supports_classification(self) -> bool:
        return self._supports_classification
        
    def supports_regression(self) -> bool:
        return self._supports_regression
        
    def requires_probabilities(self) -> bool:
        return self._requires_probabilities
        
    def is_higher_better(self) -> bool:
        return self._higher_is_better
        
    def get_value_range(self) -> tuple:
        return self._range
        
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        """Calculate the normalized mutual information score"""
        try:
            # Ensure we have discrete labels for information theory calculations
            y_true_discrete = np.asarray(y_true)
            y_pred_discrete = np.asarray(y_pred)
            
            # Calculate basic mutual information
            mi_score = mutual_info_score(y_true_discrete, y_pred_discrete)
            
            # Calculate individual entropies for normalization
            h_true = self._calculate_entropy(y_true_discrete)
            h_pred = self._calculate_entropy(y_pred_discrete)
            
            # Normalized Mutual Information using arithmetic mean normalization
            # NMI = 2 * MI(X,Y) / (H(X) + H(Y))
            if h_true + h_pred == 0:
                # Perfect agreement or both variables are constant
                nmi = 1.0 if len(np.unique(y_true_discrete)) == 1 and len(np.unique(y_pred_discrete)) == 1 else 0.0
            else:
                nmi = 2.0 * mi_score / (h_true + h_pred)
            
            # Apply information-theoretic adjustments
            nmi_adjusted = self._apply_information_adjustments(nmi, y_true_discrete, y_pred_discrete, mi_score)
            
            return float(np.clip(nmi_adjusted, 0.0, 1.0))
            
        except Exception as e:
            raise ValueError(f"Error calculating normalized mutual information: {str(e)}")
    
    def _calculate_entropy(self, labels: np.ndarray) -> float:
        """Calculate Shannon entropy of label distribution"""
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            if len(unique_labels) <= 1:
                return 0.0  # No uncertainty in constant variable
            
            # Calculate probabilities
            probabilities = counts / len(labels)
            
            # Calculate Shannon entropy: H(X) = -Σ p(x) * log2(p(x))
            shannon_entropy = entropy(probabilities, base=2)
            
            return float(shannon_entropy)
            
        except Exception:
            return 0.0
    
    def _apply_information_adjustments(self, base_nmi: float, y_true: np.ndarray, y_pred: np.ndarray, mi_score: float) -> float:
        """Apply information-theoretic adjustments to the base NMI score"""
        try:
            # Calculate number of unique labels in each variable
            n_true_labels = len(np.unique(y_true))
            n_pred_labels = len(np.unique(y_pred))
            total_samples = len(y_true)
            
            # 1. Label count balance adjustment
            # Penalize extreme differences in number of predicted vs true labels
            label_ratio = min(n_true_labels, n_pred_labels) / max(n_true_labels, n_pred_labels)
            balance_adjustment = 0.8 + 0.2 * label_ratio  # Scale from 0.8 to 1.0
            
            # 2. Sample size reliability adjustment
            # More reliable estimates with larger sample sizes
            min_samples_per_label = total_samples / max(n_true_labels, n_pred_labels)
            if min_samples_per_label >= 50:
                reliability_adjustment = 1.0
            elif min_samples_per_label >= 20:
                reliability_adjustment = 0.95
            elif min_samples_per_label >= 10:
                reliability_adjustment = 0.9
            else:
                reliability_adjustment = 0.85  # Less reliable with few samples per label
            
            # 3. Information content bonus
            # Reward high absolute mutual information (not just relative)
            max_possible_entropy = np.log2(min(n_true_labels, n_pred_labels))
            if max_possible_entropy > 0:
                info_content_ratio = mi_score / max_possible_entropy
                info_bonus = 1.0 + 0.1 * min(info_content_ratio, 1.0)  # Up to 10% bonus
            else:
                info_bonus = 1.0
            
            # 4. Clustering quality assessment
            # Check if the mutual information indicates meaningful clustering
            random_mi_baseline = self._estimate_random_mi_baseline(n_true_labels, n_pred_labels, total_samples)
            if mi_score > random_mi_baseline * 2:  # Significantly above random
                clustering_bonus = 1.05  # 5% bonus for meaningful clustering
            else:
                clustering_bonus = 1.0
            
            # Combine all adjustments
            adjusted_nmi = (base_nmi * 
                           balance_adjustment * 
                           reliability_adjustment * 
                           info_bonus * 
                           clustering_bonus)
            
            return adjusted_nmi
            
        except Exception:
            # Return base NMI if adjustments fail
            return base_nmi
    
    def _estimate_random_mi_baseline(self, n_true: int, n_pred: int, n_samples: int) -> float:
        """Estimate expected mutual information under random labeling"""
        try:
            # Simplified estimation of expected MI under independence assumption
            # This is a rough approximation for adjustment purposes
            if n_true <= 1 or n_pred <= 1 or n_samples <= 1:
                return 0.0
            
            # Expected MI under random assignment is approximately:
            # E[MI] ≈ (K1-1)(K2-1) / (2*N*ln(2)) where K1, K2 are number of clusters
            expected_random_mi = ((n_true - 1) * (n_pred - 1)) / (2 * n_samples * np.log(2))
            
            return max(0.0, expected_random_mi)
            
        except Exception:
            return 0.0
        
    def get_interpretation(self, value: float) -> str:
        """Provide interpretation of the metric value"""
        if value >= 0.9:
            return "Excellent - Very strong information agreement between true and predicted labels"
        elif value >= 0.8:
            return "Very Good - Strong mutual information indicating good clustering structure"
        elif value >= 0.7:
            return "Good - Moderate information sharing with clear patterns"
        elif value >= 0.5:
            return "Fair - Some information overlap but structure could be improved"
        elif value >= 0.3:
            return "Poor - Limited information agreement between true and predicted"
        elif value >= 0.1:
            return "Very Poor - Weak information overlap, mostly random"
        else:
            return "Random - No meaningful information sharing detected"

def get_metric_plugin() -> MetricPlugin:
    return NormalizedMutualInformationMetric()