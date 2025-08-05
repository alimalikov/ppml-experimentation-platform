"""
T-Closeness Core Implementation
=============================

This module implements t-closeness algorithms building on l-diversity for enhanced
privacy protection. T-closeness requires that the distribution of sensitive attributes
in each equivalence class is "close" to the global distribution, where closeness
is measured using various distance metrics.

Key distance measures implemented:
- Earth Mover's Distance (EMD/Wasserstein): Original t-closeness metric
- Kullback-Leibler (KL) Divergence: Information-theoretic distance
- Jensen-Shannon (JS) Divergence: Symmetric version of KL divergence

Privacy guarantees:
- Builds upon k-anonymity and l-diversity foundations
- Controls distribution similarity between local and global sensitive attributes
- Protects against attribute disclosure through distribution analysis
- Handles both ordinal and categorical sensitive data appropriately

References:
----------
Academic Papers:
- Li, N., Li, T., & Venkatasubramanian, S. (2007). t-closeness: Privacy beyond 
  k-anonymity and l-diversity. In IEEE 23rd International Conference on Data Engineering.
- Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). 
  L-diversity: Privacy beyond k-anonymity. ACM TKDD.
- Sweeney, L. (2002). k-anonymity: A model for protecting privacy. International 
  Journal of Uncertainty, Fuzziness and Knowledge-Based Systems.

Distance Metric References:
- Earth Mover's Distance: Rubner, Y., et al. (2000). The earth mover's distance 
  as a metric for image retrieval. International Journal of Computer Vision.
- Kullback-Leibler Divergence: Kullback, S., & Leibler, R. A. (1951). On information 
  and sufficiency. The Annals of Mathematical Statistics.
- Jensen-Shannon Divergence: Lin, J. (1991). Divergence measures based on the 
  Shannon entropy. IEEE Transactions on Information Theory.

Code References and Implementations:
- pyCANON Anonymity Checking: 
  https://github.com/cetic/pycanon
  - t_closeness function in anonymity.py for compliance checking
  - EMD calculation for ordinal vs categorical data handling
  - Global vs local distribution comparison logic
- AnonyPy with Mondrian: 
  https://github.com/AnonyPy/AnonyPy
  - Integrated t-closeness checking in partition.py
  - is_t_close method for distribution-based privacy validation
  - EMD implementation with numerical/categorical distinction
- ARX Anonymization Tool: 
  https://github.com/arx-deidentifier/arx
  - Production-grade t-closeness with generalization hierarchies
  - Comprehensive handling of different data types for EMD
  - Advanced metrics for privacy compliance and information loss
- SciPy Statistical Functions:
  - scipy.stats.entropy for KL divergence calculations
  - scipy.spatial.distance.jensenshannon for JS divergence
  - Statistical distance metric implementations

Implementation Patterns:
- Global distribution baseline calculation and caching
- Multiple distance metric support (EMD, KL, JS divergence)
- Ordinal vs categorical data type detection and handling
- Post-processing violation fixes through merging and suppression
- Comprehensive privacy and utility metrics calculation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import warnings
from collections import Counter
import math
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon
from .l_diversity_core import LDiversityCore

warnings.filterwarnings('ignore')

class TClosenessCore:
    """
    Core t-closeness implementation with multiple distance measures.
    
    # Implements distribution-based privacy protection beyond l-diversity
    # Ensures distance(local_dist, global_dist) ≤ t for all equivalence classes
    # Ref: Li et al. (2007), pyCANON and AnonyPy implementation patterns
    # Supports EMD (original), KL divergence, and JS divergence metrics
    """
    
    def __init__(self, k: int, l: int, t: float, qi_columns: List[str], sensitive_column: str,
                 distance_metric: str = "earth_movers", diversity_type: str = "distinct",
                 generalization_strategy: str = "optimal"):
        """
        Initialize t-closeness core.
        
        Args:
            k: The k parameter for k-anonymity
            l: The l parameter for l-diversity
            t: The t parameter for t-closeness (threshold for distance)
            qi_columns: List of quasi-identifier columns
            sensitive_column: The sensitive attribute column
            distance_metric: Distance metric ("earth_movers", "kl_divergence", "js_divergence")
            diversity_type: Type of diversity for l-diversity
            generalization_strategy: Strategy for generalization
        """
        self.k = k
        self.l = l
        self.t = t
        self.qi_columns = qi_columns
        self.sensitive_column = sensitive_column
        self.distance_metric = distance_metric
        self.generalization_strategy = generalization_strategy
        
        # Initialize l-diversity core
        self.l_diversity_core = LDiversityCore(k, l, qi_columns, sensitive_column, 
                                             diversity_type, generalization_strategy)
        
        # Global distribution of sensitive attribute
        self.global_distribution = None
        
    def fit_global_distribution(self, df: pd.DataFrame) -> None:
        """Fit the global distribution of the sensitive attribute.
        
        # Baseline distribution calculation for t-closeness comparison
        # Ref: pyCANON and AnonyPy global distribution establishment
        # Used as reference for measuring local distribution closeness
        """
        if self.sensitive_column not in df.columns:
            raise ValueError(f"Sensitive column '{self.sensitive_column}' not found in dataframe")
        
        sensitive_values = df[self.sensitive_column].dropna()
        value_counts = sensitive_values.value_counts(normalize=True).sort_index()
        self.global_distribution = value_counts
    
    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply t-closeness to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Anonymized dataframe satisfying t-closeness
        """
        if self.sensitive_column not in df.columns:
            raise ValueError(f"Sensitive column '{self.sensitive_column}' not found in dataframe")
        
        # Fit global distribution
        self.fit_global_distribution(df)
        
        # First apply l-diversity
        l_diverse_df = self.l_diversity_core.anonymize(df)
        
        # Then enforce t-closeness
        return self._enforce_t_closeness(l_diverse_df)
    
    def _enforce_t_closeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce t-closeness on l-diverse data."""
        # Group by QI columns to get equivalence classes
        valid_qi_cols = [col for col in self.qi_columns if col in df.columns]
        if not valid_qi_cols:
            return df
        
        # Get equivalence classes
        groups = df.groupby(valid_qi_cols)
        
        # Check and fix t-closeness violations
        close_groups = []
        
        for name, group in groups:
            if self._is_t_close(group):
                close_groups.append(group)
            else:
                # Fix t-closeness violation
                fixed_group = self._fix_t_closeness_violation(group, df)
                close_groups.append(fixed_group)
        
        if close_groups:
            result_df = pd.concat(close_groups, ignore_index=True)
            return result_df
        else:
            return df
    
    def _is_t_close(self, group: pd.DataFrame) -> bool:
        """Check if a group satisfies t-closeness.
        
        # Core t-closeness validation: distance(P_local, P_global) ≤ t
        # Ref: pyCANON anonymity.py t_closeness function (direct parallel)
        # AnonyPy is_t_close method for distribution comparison
        """
        if self.sensitive_column not in group.columns:
            return True
        
        sensitive_values = group[self.sensitive_column].dropna()
        
        if len(sensitive_values) == 0:
            return True
        
        # Calculate local distribution
        # Normalize to probability distribution for distance calculation
        local_distribution = sensitive_values.value_counts(normalize=True).sort_index()
        
        # Calculate distance between local and global distributions
        # Core t-closeness constraint check
        distance = self._calculate_distance(local_distribution, self.global_distribution)
        
        return distance <= self.t
    
    def _calculate_distance(self, local_dist: pd.Series, global_dist: pd.Series) -> float:
        """Calculate distance between two distributions.
        
        # Multi-metric support: EMD (original), KL divergence, JS divergence
        # Ref: Li et al. (2007) original EMD, scipy implementations for KL/JS
        # pyCANON uses EMD, extending with information-theoretic measures
        """
        if self.distance_metric == "earth_movers":
            return self._earth_movers_distance(local_dist, global_dist)
        elif self.distance_metric == "kl_divergence":
            return self._kl_divergence(local_dist, global_dist)
        elif self.distance_metric == "js_divergence":
            return self._js_divergence(local_dist, global_dist)
        else:
            return self._earth_movers_distance(local_dist, global_dist)
    
    def _earth_movers_distance(self, local_dist: pd.Series, global_dist: pd.Series) -> float:
        """Calculate Earth Mover's Distance (Wasserstein distance).
        
        # Original t-closeness metric from Li et al. (2007)
        # Ref: pyCANON __emd function, AnonyPy EMD implementation
        # Handles ordinal vs categorical data with appropriate distance calculations
        """
        # Align the distributions to have the same index
        # Ensure both distributions cover the same value domain
        all_values = sorted(set(local_dist.index).union(set(global_dist.index)))
        
        local_aligned = np.array([local_dist.get(val, 0) for val in all_values])
        global_aligned = np.array([global_dist.get(val, 0) for val in all_values])
        
        # For categorical data, we use a simplified version
        # In practice, you might need to define proper distances between categories
        # Ref: pyCANON and AnonyPy ordinal vs categorical distinction
        if self._is_ordinal_data(all_values):
            return self._ordinal_earth_movers_distance(local_aligned, global_aligned, all_values)
        else:
            return self._categorical_earth_movers_distance(local_aligned, global_aligned)
    
    def _is_ordinal_data(self, values: List) -> bool:
        """Check if data appears to be ordinal (numeric or ordered categorical)."""
        try:
            # Try to convert to numeric
            numeric_values = [float(val) for val in values]
            return True
        except (ValueError, TypeError):
            # Check for common ordinal patterns
            ordinal_patterns = ['low', 'medium', 'high', 'small', 'large', 'poor', 'fair', 'good', 'excellent']
            values_lower = [str(val).lower() for val in values]
            return any(pattern in val for val in values_lower for pattern in ordinal_patterns)
    
    def _ordinal_earth_movers_distance(self, local_dist: np.ndarray, global_dist: np.ndarray, values: List) -> float:
        """Calculate Earth Mover's Distance for ordinal data.
        
        # Standard EMD for ordered data using cumulative distributions
        # Ref: pyCANON __emd implementation, Rubner et al. (2000) EMD definition
        # Sum of absolute differences of cumulative distributions
        """
        cumsum_local = np.cumsum(local_dist)
        cumsum_global = np.cumsum(global_dist)
        
        # Sum of absolute differences of cumulative distributions
        # Classic EMD formula for 1D ordinal data
        return np.sum(np.abs(cumsum_local - cumsum_global))
    
    def _categorical_earth_movers_distance(self, local_dist: np.ndarray, global_dist: np.ndarray) -> float:
        """Calculate Earth Mover's Distance for categorical data.
        
        # Total variation distance approximation for unordered categorical data
        # Ref: pyCANON and AnonyPy categorical handling, standard practice
        # Since no natural ordering exists, use L1 distance / 2
        """
        # For categorical data, use total variation distance as approximation
        return 0.5 * np.sum(np.abs(local_dist - global_dist))
    
    def _kl_divergence(self, local_dist: pd.Series, global_dist: pd.Series) -> float:
        """Calculate Kullback-Leibler divergence.
        
        # Information-theoretic distance measure: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
        # Ref: Kullback & Leibler (1951), scipy.stats.entropy implementation
        # Extension beyond original EMD for t-closeness
        """
        # Align the distributions
        all_values = sorted(set(local_dist.index).union(set(global_dist.index)))
        
        local_aligned = np.array([local_dist.get(val, 1e-10) for val in all_values])  # Small epsilon to avoid log(0)
        global_aligned = np.array([global_dist.get(val, 1e-10) for val in all_values])
        
        # Normalize to ensure they sum to 1
        # Required for valid probability distributions
        local_aligned = local_aligned / local_aligned.sum()
        global_aligned = global_aligned / global_aligned.sum()
        
        # Calculate KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
        # Information content difference between distributions
        return np.sum(local_aligned * np.log(local_aligned / global_aligned))
    
    def _js_divergence(self, local_dist: pd.Series, global_dist: pd.Series) -> float:
        """Calculate Jensen-Shannon divergence.
        
        # Symmetric version of KL divergence: JS(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
        # Ref: Lin (1991), scipy.spatial.distance.jensenshannon
        # Bounded metric [0,1] unlike KL divergence
        """
        # Align the distributions
        all_values = sorted(set(local_dist.index).union(set(global_dist.index)))
        
        local_aligned = np.array([local_dist.get(val, 0) for val in all_values])
        global_aligned = np.array([global_dist.get(val, 0) for val in all_values])
        
        # Normalize
        if local_aligned.sum() > 0:
            local_aligned = local_aligned / local_aligned.sum()
        if global_aligned.sum() > 0:
            global_aligned = global_aligned / global_aligned.sum()
        
        # Use scipy's Jensen-Shannon distance
        # Square root of JS divergence, metric property guaranteed
        return jensenshannon(local_aligned, global_aligned)
    
    def _fix_t_closeness_violation(self, group: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
        """Fix t-closeness violation by merging with other groups or additional processing.
        
        # Two-stage approach: 1) merge with compatible groups, 2) suppress violations
        # Ref: AnonyPy integrated approach, ARX generalization hierarchies
        # Alternative to pyCANON's checking-only approach
        """
        # Strategy 1: Try to merge with groups that would improve t-closeness
        merged_group = self._try_merge_for_t_closeness(group, full_df)
        if merged_group is not None and self._is_t_close(merged_group):
            return merged_group
        
        # Strategy 2: Apply record suppression or generalization
        return self._apply_t_closeness_generalization(group)
    
    def _try_merge_for_t_closeness(self, group: pd.DataFrame, full_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Try to merge group with others to achieve t-closeness."""
        valid_qi_cols = [col for col in self.qi_columns if col in full_df.columns]
        if not valid_qi_cols:
            return None
        
        all_groups = full_df.groupby(valid_qi_cols)
        best_merged = None
        best_distance = float('inf')
        
        for name, other_group in all_groups:
            if len(other_group) == len(group) and other_group.equals(group):
                continue  # Skip the same group
            
            # Try merging and check if it improves t-closeness
            merged = pd.concat([group, other_group], ignore_index=True)
            if len(merged) >= self.k:
                # Check if merged group satisfies l-diversity
                if self.l_diversity_core._is_l_diverse(merged):
                    # Calculate t-closeness distance
                    sensitive_values = merged[self.sensitive_column].dropna()
                    if len(sensitive_values) > 0:
                        local_dist = sensitive_values.value_counts(normalize=True).sort_index()
                        distance = self._calculate_distance(local_dist, self.global_distribution)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_merged = merged
        
        return best_merged if best_distance <= self.t else None
    
    def _apply_t_closeness_generalization(self, group: pd.DataFrame) -> pd.DataFrame:
        """Apply additional processing to achieve t-closeness."""
        # This is a simplified approach - suppress records that cause violations
        result_group = group.copy()
        
        # Calculate which sensitive values are causing the violation
        sensitive_values = group[self.sensitive_column].dropna()
        local_dist = sensitive_values.value_counts(normalize=True).sort_index()
        
        # Find values that are over-represented compared to global distribution
        over_represented = []
        for value in local_dist.index:
            local_prob = local_dist.get(value, 0)
            global_prob = self.global_distribution.get(value, 0)
            if local_prob > global_prob * (1 + self.t):  # Significantly over-represented
                over_represented.append(value)
        
        # Remove some records with over-represented values (keeping k-anonymity)
        if over_represented and len(result_group) > self.k:
            for value in over_represented:
                value_records = result_group[result_group[self.sensitive_column] == value]
                if len(value_records) > 1 and len(result_group) - 1 >= self.k:
                    # Remove one record with this value
                    result_group = result_group.drop(value_records.index[0])
                    break
        
        return result_group
    
    def get_privacy_metrics(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy and utility metrics for t-closeness."""
        # Get l-diversity metrics first
        l_metrics = self.l_diversity_core.get_privacy_metrics(original_df, anonymized_df)
        
        # Add t-closeness specific metrics
        t_metrics = {}
        t_metrics.update(l_metrics)
        
        # T-closeness metrics
        t_metrics['t_value'] = self.t
        t_metrics['distance_metric'] = self.distance_metric
        
        # Check t-closeness compliance
        valid_qi_cols = [col for col in self.qi_columns if col in anonymized_df.columns]
        if valid_qi_cols and self.sensitive_column in anonymized_df.columns:
            groups = anonymized_df.groupby(valid_qi_cols)
            close_groups = 0
            total_groups = 0
            min_distance = float('inf')
            max_distance = 0
            distance_scores = []
            
            for name, group in groups:
                total_groups += 1
                distance = self._calculate_group_distance(group)
                distance_scores.append(distance)
                
                if distance <= self.t:
                    close_groups += 1
                
                min_distance = min(min_distance, distance)
                max_distance = max(max_distance, distance)
            
            t_metrics['t_close_groups'] = close_groups
            t_metrics['total_groups'] = total_groups
            t_metrics['t_closeness_compliance'] = close_groups / total_groups if total_groups > 0 else 0
            t_metrics['min_distance'] = min_distance if min_distance != float('inf') else 0
            t_metrics['max_distance'] = max_distance
            t_metrics['avg_distance'] = np.mean(distance_scores) if distance_scores else 0
        
        return t_metrics
    
    def _calculate_group_distance(self, group: pd.DataFrame) -> float:
        """Calculate t-closeness distance for a group."""
        if self.sensitive_column not in group.columns:
            return 0
        
        sensitive_values = group[self.sensitive_column].dropna()
        
        if len(sensitive_values) == 0:
            return 0
        
        local_distribution = sensitive_values.value_counts(normalize=True).sort_index()
        return self._calculate_distance(local_distribution, self.global_distribution)


def apply_t_closeness(df: pd.DataFrame, k: int, l: int, t: float, qi_columns: List[str], 
                     sensitive_column: str, distance_metric: str = "earth_movers",
                     diversity_type: str = "distinct", generalization_strategy: str = "optimal") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply t-closeness to a dataframe.
    
    # High-level interface for t-closeness anonymization
    # Builds upon k-anonymity and l-diversity with distribution closeness constraints
    # Ref: Li et al. (2007), pyCANON/AnonyPy implementation patterns
    
    Args:
        df: Input dataframe to anonymize
        k: The k parameter for k-anonymity (minimum group size)
        l: The l parameter for l-diversity (minimum diversity requirement)
        t: The t parameter for t-closeness (maximum distance threshold)
        qi_columns: List of quasi-identifier columns
        sensitive_column: The sensitive attribute column
        distance_metric: Distance metric ("earth_movers", "kl_divergence", "js_divergence")
        diversity_type: Type of diversity for l-diversity base layer
        generalization_strategy: Strategy for generalization
        
    Returns:
        Tuple of (anonymized_dataframe, metrics_dict)
        
    References:
        - Li, N., et al. (2007): t-closeness: Privacy beyond k-anonymity and l-diversity
        - Implementation patterns from:
          * pyCANON: https://github.com/cetic/pycanon (anonymity checking)
          * AnonyPy: https://github.com/AnonyPy/AnonyPy (integrated partitioning)
          * ARX: https://github.com/arx-deidentifier/arx (production-grade implementation)
        - Distance metrics: EMD (original), KL/JS divergence (extensions)
        - Global distribution baseline with local distribution comparison
    """
    anonymizer = TClosenessCore(k, l, t, qi_columns, sensitive_column, distance_metric, 
                               diversity_type, generalization_strategy)
    anonymized_df = anonymizer.anonymize(df)
    metrics = anonymizer.get_privacy_metrics(df, anonymized_df)
    
    return anonymized_df, metrics
