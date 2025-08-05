"""
Differential Privacy Core Implementation
======================================

This module implements differential privacy mechanisms for data anonymization.
Differential privacy provides formal privacy guarantees by adding calibrated noise
to data or query results.

Key mechanisms implemented:
- Laplace mechanism for continuous data
- Exponential mechanism for categorical data
- Gaussian mechanism for improved utility
- Composition tracking for privacy budget management

Privacy Parameters:
- epsilon (ε): Privacy budget - smaller values = stronger privacy
- delta (δ): Probability of privacy breach - typically very small
- sensitivity: Maximum change in output when one record changes

References:
----------
Academic Papers:
- Dwork, C. (2006). Differential privacy. In International colloquium on automata, 
  languages, and programming (pp. 1-12). Springer.
- Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. 
  Foundations and Trends in Theoretical Computer Science, 9(3-4), 211-407.
- McSherry, F., & Talwar, K. (2007). Mechanism design via differential privacy. 
  In 48th Annual IEEE Symposium on Foundations of Computer Science (pp. 94-103).
- Balle, B., & Wang, Y. X. (2018). Improving the gaussian mechanism for differential 
  privacy: Analytical calibration and optimal denoising. In International Conference 
  on Machine Learning (pp. 394-403).
- Warner, S. L. (1965). Randomized response: A survey technique for eliminating 
  evasive answer bias. Journal of the American Statistical Association, 60(309), 63-69.

Code References and Implementations:
- IBM Differential Privacy Library (diffprivlib): 
  https://github.com/IBM/differential-privacy-library
  - Laplace mechanism: diffprivlib/mechanisms/laplace.py
  - Gaussian mechanism: diffprivlib/mechanisms/gaussian.py
- Google Differential Privacy Library: 
  https://github.com/google/differential-privacy
  - Secure noise generation for Laplace and Gaussian mechanisms
- OpenMined PyDP (Python wrapper for Google's DP): 
  https://github.com/OpenMined/PyDP
  - BoundedMean algorithm implementation
- Analytic Gaussian Mechanism: 
  https://github.com/BorjaBalle/analytic-gaussian-mechanism
  - Implementation based on Balle & Wang (2018)
- Exponential Mechanism (Base-2): 
  https://github.com/cilvento/b2_exponential_mechanism
  - High precision exponential mechanism implementation
- Practical Data Privacy: 
  https://github.com/kjam/practical-data-privacy
  - Randomized response implementation examples
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class DifferentialPrivacyCore:
    """
    Core implementation of differential privacy mechanisms.
    
    This class provides methods for applying differential privacy to datasets
    using various noise mechanisms while tracking privacy budget consumption.
    """
    
    def __init__(self):
        self.privacy_budget_used = 0.0
        self.composition_method = "basic"  # "basic", "advanced", "rdp"
        self.noise_history = []
        
    def reset_privacy_budget(self):
        """Reset the privacy budget tracker."""
        self.privacy_budget_used = 0.0
        self.noise_history = []
        
    def get_remaining_budget(self, total_epsilon: float) -> float:
        """Get remaining privacy budget."""
        return max(0.0, total_epsilon - self.privacy_budget_used)
        
    def validate_privacy_parameters(self, epsilon: float, delta: float = None) -> bool:
        """
        Validate differential privacy parameters.
        
        Args:
            epsilon: Privacy parameter (> 0)
            delta: Privacy parameter (0 <= delta <= 1) for approximate DP
            
        Returns:
            bool: True if parameters are valid
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
            
        if delta is not None:
            if not (0 <= delta <= 1):
                raise ValueError("Delta must be between 0 and 1")
                
        return True
        
    def calculate_global_sensitivity(self, df: pd.DataFrame, columns: List[str], 
                                   operation: str = "mean") -> Dict[str, float]:
        """
        Calculate global sensitivity for numerical columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to calculate sensitivity for
            operation: Type of operation ("mean", "sum", "count", "max", "min")
            
        Returns:
            Dict mapping column names to their sensitivities
        """
        sensitivities = {}
        n = len(df)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            col_data = df[col].dropna()
            if len(col_data) == 0:
                sensitivities[col] = 1.0
                continue
                
            if operation == "mean":
                # For mean: sensitivity = range / n
                col_range = col_data.max() - col_data.min()
                sensitivities[col] = col_range / n if n > 0 else 1.0
                
            elif operation == "sum":
                # For sum: sensitivity = max absolute value
                sensitivities[col] = col_data.abs().max()
                
            elif operation == "count":
                # For count: sensitivity = 1 (one record can change count by 1)
                sensitivities[col] = 1.0
                
            elif operation in ["max", "min"]:
                # For max/min: sensitivity = range
                sensitivities[col] = col_data.max() - col_data.min()
                
            else:
                # Default sensitivity
                sensitivities[col] = col_data.std() if col_data.std() > 0 else 1.0
                
        return sensitivities
        
    def laplace_mechanism(self, data: Union[pd.Series, np.ndarray, float], 
                         sensitivity: float, epsilon: float) -> Union[pd.Series, np.ndarray, float]:
        """
        Apply Laplace mechanism for differential privacy.
        
        The Laplace mechanism adds noise drawn from the Laplace distribution
        to achieve ε-differential privacy. The noise scale is calibrated to
        the sensitivity of the function and the privacy parameter ε.
        
        Args:
            data: Input data (can be Series, array, or scalar)
            sensitivity: Global sensitivity of the function
            epsilon: Privacy parameter
            
        Returns:
            Data with Laplace noise added
            
        References:
            - Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). 
              Calibrating noise to sensitivity in private data analysis.
            - IBM diffprivlib implementation: 
              https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/laplace.py
            - Google DP library: 
              https://github.com/google/differential-privacy
        """
        self.validate_privacy_parameters(epsilon)
        
        # Calculate noise scale
        scale = sensitivity / epsilon
        
        if isinstance(data, pd.Series):
            noise = np.random.laplace(0, scale, size=len(data))
            result = data + noise
            self.noise_history.append({
                'mechanism': 'laplace',
                'scale': scale,
                'epsilon_used': epsilon,
                'size': len(data)
            })
            return result
            
        elif isinstance(data, np.ndarray):
            noise = np.random.laplace(0, scale, size=data.shape)
            result = data + noise
            self.noise_history.append({
                'mechanism': 'laplace',
                'scale': scale,
                'epsilon_used': epsilon,
                'size': data.size
            })
            return result
            
        else:  # scalar
            noise = np.random.laplace(0, scale)
            result = data + noise
            self.noise_history.append({
                'mechanism': 'laplace',
                'scale': scale,
                'epsilon_used': epsilon,
                'size': 1
            })
            return result
            
    def gaussian_mechanism(self, data: Union[pd.Series, np.ndarray, float], 
                          sensitivity: float, epsilon: float, delta: float) -> Union[pd.Series, np.ndarray, float]:
        """
        Apply Gaussian mechanism for (ε,δ)-differential privacy.
        
        The Gaussian mechanism provides (ε,δ)-differential privacy by adding
        Gaussian noise with variance calibrated to the sensitivity and privacy
        parameters. Often provides better utility than Laplace mechanism.
        
        Args:
            data: Input data
            sensitivity: Global sensitivity
            epsilon: Privacy parameter
            delta: Privacy parameter
            
        Returns:
            Data with Gaussian noise added
            
        References:
            - Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy.
            - Balle, B., & Wang, Y. X. (2018). Improving the gaussian mechanism for 
              differential privacy: Analytical calibration and optimal denoising.
            - IBM diffprivlib Gaussian mechanism: 
              https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/gaussian.py
            - Analytic Gaussian Mechanism: 
              https://github.com/BorjaBalle/analytic-gaussian-mechanism
        """
        self.validate_privacy_parameters(epsilon, delta)
        
        # Calculate noise scale for Gaussian mechanism
        # σ ≥ sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        if isinstance(data, pd.Series):
            noise = np.random.normal(0, sigma, size=len(data))
            result = data + noise
            self.noise_history.append({
                'mechanism': 'gaussian',
                'sigma': sigma,
                'epsilon_used': epsilon,
                'delta_used': delta,
                'size': len(data)
            })
            return result
            
        elif isinstance(data, np.ndarray):
            noise = np.random.normal(0, sigma, size=data.shape)
            result = data + noise
            self.noise_history.append({
                'mechanism': 'gaussian',
                'sigma': sigma,
                'epsilon_used': epsilon,
                'delta_used': delta,
                'size': data.size
            })
            return result
            
        else:  # scalar
            noise = np.random.normal(0, sigma)
            result = data + noise
            self.noise_history.append({
                'mechanism': 'gaussian',
                'sigma': sigma,
                'epsilon_used': epsilon,
                'delta_used': delta,
                'size': 1
            })
            return result
            
    def exponential_mechanism(self, candidates: List[Any], utility_scores: List[float], 
                            sensitivity: float, epsilon: float) -> Any:
        """
        Apply exponential mechanism for selecting from discrete candidates.
        
        The exponential mechanism selects an output from a discrete set of candidates
        with probability proportional to the exponential of their utility scores.
        Useful for categorical data and discrete optimization problems.
        
        Args:
            candidates: List of candidate values
            utility_scores: Utility score for each candidate
            sensitivity: Sensitivity of utility function
            epsilon: Privacy parameter
            
        Returns:
            Selected candidate
            
        References:
            - McSherry, F., & Talwar, K. (2007). Mechanism design via differential privacy.
            - Exponential Mechanism (Base-2): 
              https://github.com/cilvento/b2_exponential_mechanism
            - IBM diffprivlib exponential mechanism: 
              https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/exponential.py
        """
        self.validate_privacy_parameters(epsilon)
        
        if len(candidates) != len(utility_scores):
            raise ValueError("Candidates and utility scores must have same length")
            
        # Calculate probabilities using exponential mechanism
        scaled_utilities = np.array(utility_scores) * epsilon / (2 * sensitivity)
        
        # Subtract max for numerical stability
        scaled_utilities = scaled_utilities - np.max(scaled_utilities)
        
        # Calculate probabilities
        probabilities = np.exp(scaled_utilities)
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample according to probabilities
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        
        self.noise_history.append({
            'mechanism': 'exponential',
            'epsilon_used': epsilon,
            'num_candidates': len(candidates),
            'selected_utility': utility_scores[selected_idx]
        })
        
        return candidates[selected_idx]
        
    def randomized_response(self, data: pd.Series, epsilon: float, 
                           possible_values: List[Any] = None) -> pd.Series:
        """
        Apply randomized response mechanism for categorical data.
        
        Randomized response is a technique for collecting sensitive information
        while preserving plausible deniability. Each respondent has a probability
        of giving their true answer and a probability of giving a random answer.
        
        Args:
            data: Input categorical data
            epsilon: Privacy parameter
            possible_values: List of possible values (if None, inferred from data)
            
        Returns:
            Perturbed categorical data
            
        References:
            - Warner, S. L. (1965). Randomized response: A survey technique for 
              eliminating evasive answer bias. Journal of the American Statistical Association.
            - Practical Data Privacy: 
              https://github.com/kjam/practical-data-privacy
            - OpenMined tutorial on randomized response: 
              https://blog.openmined.org/randomized-response-in-privacy/
            - Differential Privacy for Categorical Data: 
              https://github.com/llgeek/K-anonymity-and-Differential-Privacy
        """
        self.validate_privacy_parameters(epsilon)
        
        if possible_values is None:
            possible_values = list(data.unique())
            
        k = len(possible_values)
        
        # Calculate probabilities for randomized response
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)  # Probability of truth
        q = 1 / (np.exp(epsilon) + k - 1)  # Probability of each other value
        
        result = []
        for value in data:
            if np.random.random() < p:
                # Output true value
                result.append(value)
            else:
                # Output random value from domain
                other_values = [v for v in possible_values if v != value]
                if other_values:
                    result.append(np.random.choice(other_values))
                else:
                    result.append(value)
                    
        self.noise_history.append({
            'mechanism': 'randomized_response',
            'epsilon_used': epsilon,
            'domain_size': k,
            'truth_probability': p
        })
        
        return pd.Series(result, index=data.index)
        
    def apply_differential_privacy(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply differential privacy to a DataFrame based on configuration.
        
        Args:
            df: Input DataFrame
            config: Configuration dictionary with privacy parameters
            
        Returns:
            DataFrame with differential privacy applied
        """
        result_df = df.copy()
        
        # Extract configuration parameters
        epsilon = config.get('epsilon', 1.0)
        delta = config.get('delta', 1e-5)
        mechanism = config.get('mechanism', 'laplace')
        columns_to_anonymize = config.get('columns', df.columns.tolist())
        operation = config.get('operation', 'identity')
        
        # Track privacy budget usage
        epsilon_per_column = epsilon / len(columns_to_anonymize) if columns_to_anonymize else epsilon
        
        # Calculate sensitivities
        numeric_columns = [col for col in columns_to_anonymize 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        categorical_columns = [col for col in columns_to_anonymize 
                             if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_columns:
            sensitivities = self.calculate_global_sensitivity(df, numeric_columns, operation)
            
            # Apply noise to numeric columns
            for col in numeric_columns:
                if col not in sensitivities:
                    continue
                    
                sensitivity = sensitivities[col]
                
                if mechanism == 'laplace':
                    result_df[col] = self.laplace_mechanism(
                        df[col], sensitivity, epsilon_per_column
                    )
                elif mechanism == 'gaussian':
                    result_df[col] = self.gaussian_mechanism(
                        df[col], sensitivity, epsilon_per_column, delta
                    )
                    
        # Apply randomized response to categorical columns
        if categorical_columns:
            epsilon_per_cat_column = epsilon_per_column
            
            for col in categorical_columns:
                result_df[col] = self.randomized_response(
                    df[col], epsilon_per_cat_column
                )
                
        # Update privacy budget
        self.privacy_budget_used += epsilon
        
        return result_df
        
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """
        Get privacy-related metrics and information.
        
        Returns:
            Dictionary with privacy metrics
        """
        return {
            'total_epsilon_used': self.privacy_budget_used,
            'number_of_operations': len(self.noise_history),
            'mechanisms_used': list(set(h.get('mechanism', 'unknown') for h in self.noise_history)),
            'composition_method': self.composition_method,
            'noise_history': self.noise_history.copy()
        }
        
    def estimate_privacy_loss(self, config: Dict[str, Any], df_size: int) -> Dict[str, float]:
        """
        Estimate privacy loss for a given configuration.
        
        Args:
            config: Privacy configuration
            df_size: Size of dataset
            
        Returns:
            Dictionary with privacy loss estimates
        """
        epsilon = config.get('epsilon', 1.0)
        delta = config.get('delta', 1e-5)
        columns = config.get('columns', [])
        
        # Basic composition (can be improved with advanced composition)
        total_epsilon = epsilon * len(columns) if columns else epsilon
        
        return {
            'estimated_epsilon': total_epsilon,
            'estimated_delta': delta,
            'privacy_level': self._classify_privacy_level(total_epsilon),
            'dataset_size': df_size,
            'relative_privacy_cost': total_epsilon / df_size if df_size > 0 else float('inf')
        }
        
    def _classify_privacy_level(self, epsilon: float) -> str:
        """Classify privacy level based on epsilon value."""
        if epsilon <= 0.1:
            return "Very High Privacy"
        elif epsilon <= 0.5:
            return "High Privacy"
        elif epsilon <= 1.0:
            return "Moderate Privacy"
        elif epsilon <= 2.0:
            return "Low Privacy"
        else:
            return "Very Low Privacy"
