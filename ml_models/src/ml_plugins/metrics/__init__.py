"""Metric plugins package."""

# Import metric plugins
try:
    from .roc_auc_metric import ROCAUCMetric
    from .matthews_correlation_metric import MatthewsCorrelationMetric
    # from .precision_recall_auc_metric import PrecisionRecallAUCMetric
    # from .balanced_accuracy_metric import BalancedAccuracyMetric
    # from .cohen_kappa_metric import CohenKappaMetric
    from .log_loss_metric import LogLossMetric
    # from .mean_squared_error_metric import MeanSquaredErrorMetric
    # from .mean_absolute_error_metric import MeanAbsoluteErrorMetric
    # from .r2_score_metric import R2ScoreMetric
    # from .adjusted_r2_metric import AdjustedR2Metric
    
    __all__ = [
        'ROCAUCMetric',
        'MatthewsCorrelationMetric', 
        'PrecisionRecallAUCMetric',
        'BalancedAccuracyMetric',
        'CohenKappaMetric',
        'LogLossMetric',
        'MeanSquaredErrorMetric',
        'MeanAbsoluteErrorMetric',
        'R2ScoreMetric',
        'AdjustedR2Metric'
    ]
except ImportError as e:
    print(f"Error importing metric plugins: {e}")
    __all__ = []