import numpy as np
from typing import Optional
from src.ml_plugins.base_metric_plugin import MetricPlugin

class ClassificationAccuracyMetric(MetricPlugin):
    """Classification Accuracy per Class metric for detailed multi-class analysis."""
    
    def __init__(self):
        super().__init__()
        self._name = "Per-Class Accuracy"
        self._description = "Average of per-class accuracy scores. Shows how well the model performs on each individual class."
        self._category = "Classification"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = False
        self._higher_is_better = True
        self._range = (0, 1)
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            # Get unique classes
            classes = np.unique(y_true)
            
            if len(classes) == 0:
                return 0.0
            
            class_accuracies = []
            
            for cls in classes:
                # For each class, calculate accuracy as: correct predictions for this class / total instances of this class
                cls_mask = (y_true == cls)
                if np.sum(cls_mask) == 0:
                    continue
                
                cls_accuracy = np.sum((y_pred == cls) & cls_mask) / np.sum(cls_mask)
                class_accuracies.append(cls_accuracy)
            
            # Return average of per-class accuracies
            if len(class_accuracies) == 0:
                return 0.0
            
            return float(np.mean(class_accuracies))
        except Exception as e:
            raise ValueError(f"Error calculating Per-Class Accuracy: {str(e)}")
    
    def get_category(self) -> str:
        return self._category
    
    def supports_classification(self) -> bool:
        return self._supports_classification
    
    def supports_regression(self) -> bool:
        return self._supports_regression
    
    def requires_probabilities(self) -> bool:
        return self._requires_probabilities
    
    def higher_is_better(self) -> bool:
        return self._higher_is_better
    
    def get_range(self) -> tuple:
        return self._range


def get_metric_plugin() -> MetricPlugin:
    return ClassificationAccuracyMetric()
