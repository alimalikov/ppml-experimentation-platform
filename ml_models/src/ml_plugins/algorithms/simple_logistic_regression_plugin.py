import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler

# For test environment, use relative imports based on temp_plugins depth
from src.ml_plugins.base_ml_plugin import MLPlugin

class SimpleLogisticRegression(BaseEstimator, ClassifierMixin, MLPlugin):
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self._name = "Simple Logistic Regression"
        self._description = "A simple logistic regression implementation with gradient descent"
        self._category = "Linear Models"
        # Required capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._min_samples_required = 50

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return self._description

    def get_category(self) -> str:
        return self._category

    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # Handle multi-class by converting to binary (vs rest)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            # Binary classification
            y_binary = (y == self.classes_[1]).astype(int)
            # Standardize features
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            # Add bias term
            X_with_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
            # Initialize weights
            self.weights_ = np.random.normal(0, 0.01, X_with_bias.shape[1])
            # Gradient descent
            for iteration in range(self.max_iterations):
                # Forward pass
                z = X_with_bias.dot(self.weights_)
                predictions = self._sigmoid(z)
                # Compute cost with regularization
                cost = -np.mean(y_binary * np.log(predictions + 1e-15) +
                               (1 - y_binary) * np.log(1 - predictions + 1e-15))
                cost += self.regularization * np.sum(self.weights_[1:] ** 2)
                # Compute gradients
                gradients = X_with_bias.T.dot(predictions - y_binary) / len(y_binary)
                gradients[1:] += 2 * self.regularization * self.weights_[1:]
                # Update weights
                self.weights_ -= self.learning_rate * gradients
                # Early stopping if cost is very small
                if cost < 1e-6:
                    break
        else:
            # Multi-class: use one-vs-rest
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            X_with_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
            self.weights_ = []
            for class_label in self.classes_:
                y_binary = (y == class_label).astype(int)
                # Initialize weights for this class
                weights = np.random.normal(0, 0.01, X_with_bias.shape[1])
                # Train binary classifier
                for iteration in range(self.max_iterations):
                    z = X_with_bias.dot(weights)
                    predictions = self._sigmoid(z)
                    cost = -np.mean(y_binary * np.log(predictions + 1e-15) +
                                   (1 - y_binary) * np.log(1 - predictions + 1e-15))
                    cost += self.regularization * np.sum(weights[1:] ** 2)
                    gradients = X_with_bias.T.dot(predictions - y_binary) / len(y_binary)
                    gradients[1:] += 2 * self.regularization * weights[1:]
                    weights -= self.learning_rate * gradients
                    if cost < 1e-6:
                        break
                self.weights_.append(weights)
            self.weights_ = np.array(self.weights_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        X_scaled = self.scaler_.transform(X)
        X_with_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        if len(self.classes_) == 2:
            # Binary classification
            z = X_with_bias.dot(self.weights_)
            probabilities = self._sigmoid(z)
            predictions = (probabilities >= 0.5).astype(int)
            return np.array([self.classes_[pred] for pred in predictions])
        else:
            # Multi-class classification
            class_scores = []
            for weights in self.weights_:
                z = X_with_bias.dot(weights)
                scores = self._sigmoid(z)
                class_scores.append(scores)
            class_scores = np.array(class_scores).T
            predicted_indices = np.argmax(class_scores, axis=1)
            return self.classes_[predicted_indices]

    def predict_proba(self, X):
        X = check_array(X)
        X_scaled = self.scaler_.transform(X)
        X_with_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        if len(self.classes_) == 2:
            # Binary classification
            z = X_with_bias.dot(self.weights_)
            prob_class_1 = self._sigmoid(z)
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            # Multi-class classification
            class_scores = []
            for weights in self.weights_:
                z = X_with_bias.dot(weights)
                scores = self._sigmoid(z)
                class_scores.append(scores)
            class_scores = np.array(class_scores).T
            # Normalize to get probabilities
            probabilities = class_scores / np.sum(class_scores, axis=1, keepdims=True)
            return probabilities

    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        st.sidebar.subheader(f"{self.get_name()} Configuration")
        learning_rate = st.sidebar.number_input(
            "Learning Rate:",
            value=self.learning_rate,
            min_value=0.001,
            max_value=1.0,
            step=0.001,
            key=f"{key_prefix}_learning_rate",
            help="How fast the algorithm learns (smaller = more stable)"
        )
        max_iterations = st.sidebar.number_input(
            "Max Iterations:",
            value=self.max_iterations,
            min_value=100,
            max_value=5000,
            step=100,
            key=f"{key_prefix}_max_iterations",
            help="Maximum number of training iterations"
        )
        regularization = st.sidebar.number_input(
            "Regularization:",
            value=self.regularization,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            key=f"{key_prefix}_regularization",
            help="Prevents overfitting (higher = more regularization)"
        )
        return {
            "learning_rate": learning_rate,
            "max_iterations": max_iterations,
            "regularization": regularization
        }

    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return SimpleLogisticRegression(
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            max_iterations=hyperparameters.get("max_iterations", self.max_iterations),
            regularization=hyperparameters.get("regularization", self.regularization)
        )

    def preprocess_data(self, X, y):
        """Optional data preprocessing"""
        return X, y

    def is_compatible_with_data(self, df, target_column):
        """Check if algorithm is compatible with the data"""
        try:
            # Check if target is suitable for classification
            target_series = df[target_column]
            # Check for reasonable number of classes
            n_unique = target_series.nunique()
            if n_unique < 2:
                return False, "Need at least 2 classes for classification"
            elif n_unique > 20:
                return False, "Too many classes (>20), consider using a different algorithm"
            # Check for sufficient data
            if len(df) < 50:
                return False, "Need at least 50 samples for reliable training"
            # Check for reasonable feature count
            n_features = len(df.columns) - 1  # Subtract target column
            if n_features < 1:
                return False, "Need at least 1 feature for training"
            elif n_features > len(df) / 2:
                return False, "Too many features relative to samples (risk of overfitting)"
            return True, "Compatible"
        except Exception as e:
            return False, str(e)

def get_plugin():
    return SimpleLogisticRegression()
