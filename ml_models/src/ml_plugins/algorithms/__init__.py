"""ML Algorithm plugins package."""

# Import algorithm plugins
try:
    from .logistic_regression_classifier_plugin import *
    from .random_forest_classifier_plugin import *
    from .xgboost_classifier_plugin import *
    from .lightbgm_classifier_plugin import *
    from .decision_tree_classifier_plugin import *
    from .elastic_net_classifier_plugin import *
    from .extra_trees_classifier_plugin import *
    from .linear_svm_plugin import *
    from .perceptron_plugin import *
    from .ridge_classifier_plugin import *
    
    __all__ = []
except ImportError as e:
    print(f"Error importing algorithm plugins: {e}")
    __all__ = []