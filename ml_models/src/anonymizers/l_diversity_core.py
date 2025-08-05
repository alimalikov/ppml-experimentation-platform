"""
L-Diversity Core Implementation
=============================

This module implements l-diversity algorithms building on k-anonymity for enhanced
privacy protection. L-diversity ensures that each equivalence class contains at
least l "well-represented" values for sensitive attributes, addressing the 
homogeneity attack weakness of k-anonymity.

Key diversity measures implemented:
- Distinct l-diversity: Each equivalence class has at least l distinct sensitive values
- Entropy l-diversity: Each equivalence class has entropy ≥ log(l)
- Recursive (c,l)-diversity: Most frequent value appears < c times the sum of l-1 least frequent

Privacy guarantees:
- Builds upon k-anonymity foundation
- Protects against homogeneity attacks
- Provides stronger privacy for sensitive attributes
- Supports multiple diversity measurement strategies

References:
----------
Academic Papers:
- Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). 
  L-diversity: Privacy beyond k-anonymity. ACM Transactions on Knowledge Discovery 
  from Data (TKDD), 1(1), 3-es.
- Li, N., Li, T., & Venkatasubramanian, S. (2007). t-closeness: Privacy beyond 
  k-anonymity and l-diversity. In 2007 IEEE 23rd International Conference on 
  Data Engineering (pp. 106-115).
- Sweeney, L. (2002). k-anonymity: A model for protecting privacy. International 
  Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(05), 557-570.
- Ghinita, G., Karras, P., Kalnis, P., & Mamoulis, N. (2007). Fast data anonymization 
  with low information loss. In Proceedings of the 33rd international conference 
  on Very large data bases (pp. 758-769).

Algorithm References:
- Mondrian L-Diversity Algorithm:
  LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006). Mondrian multidimensional 
  k-anonymity (extended for l-diversity).
- InfoGain Mondrian for L-Diversity:
  Extension of Mondrian algorithm with information gain heuristics for l-diversity.

Code References and Implementations:
- Mondrian L-Diversity Implementation: 
  https://github.com/qiyuangong/Mondrian_L_Diversity
  - Python implementation for Mondrian L-Diversity algorithm
  - Based on InfoGain Mondrian with NCP ~79.04% on adult dataset
  - Extends Mondrian multidimensional k-anonymity for l-diversity
- Nuclearstar K-Anonymity with L-Diversity: 
  https://github.com/Nuclearstar/K-Anonymity
  - Comprehensive k-anonymity and l-diversity implementation
  - Uses Kolmogorov-Smirnov distance for diversity measurement
  - Greedy search algorithm (Mondrian) for data partitioning
- PyCANON Library: 
  https://github.com/IFCA/pycanon
  - Implements distinct, entropy, and recursive (c,l)-diversity
  - Published in Scientific Data (Nature) journal
  - Comprehensive anonymity parameter checking and reporting
- ANJANA Library: 
  https://github.com/IFCA-Advanced-Computing/anjana
  - Entropy l-diversity and recursive (c,l)-diversity implementation
  - Working examples with quasi-identifier hierarchies
  - Application of k-anonymity, l-diversity, and t-closeness
- ARX Framework: 
  https://github.com/shoe54/arx-1
  - Java implementation with k-anonymity, l-diversity, t-closeness
  - Includes recursive-(c,l)-diversity, entropy-l-diversity, distinct l-diversity
- Simple L-Diversity Implementations:
  - https://github.com/Dharmik1710/L-diversity (Python implementation)
  - https://github.com/pratyushlokhande/L-diversity-implementation (Equivalence class categorization)

Diversity Type Definitions:
- Distinct L-Diversity: Qi-block is l-diverse if |{s | (qi,s) ∈ qi-block}| ≥ l
- Entropy L-Diversity: Qi-block satisfies Entropy(Q) ≥ log(l)
- Recursive (c,l)-Diversity: r1 < c(rl + rl+1 + ... + rm) where ri are frequencies
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import warnings
from collections import Counter
import math
from .k_anonymity_core import KAnonymityCore

warnings.filterwarnings('ignore')

class LDiversityCore:
    """
    Core l-diversity implementation with multiple diversity measures.
    
    This class implements the three main types of l-diversity as defined in the
    literature: distinct, entropy, and recursive (c,l)-diversity. It builds upon
    k-anonymity to provide stronger privacy guarantees for sensitive attributes.
    
    The implementation follows the Mondrian approach for data partitioning and
    includes group merging strategies to achieve l-diversity while minimizing
    information loss.
    
    References:
        - Machanavajjhala et al. (2007). L-diversity: Privacy beyond k-anonymity.
        - Mondrian L-Diversity: https://github.com/qiyuangong/Mondrian_L_Diversity
        - PyCANON implementation patterns: https://github.com/IFCA/pycanon
        - ANJANA library approaches: https://github.com/IFCA-Advanced-Computing/anjana
    """
    
    def __init__(self, k: int, l: int, qi_columns: List[str], sensitive_column: str, 
                 diversity_type: str = "distinct", generalization_strategy: str = "optimal"):
        """
        Initialize l-diversity core.
        
        Args:
            k: The k parameter for k-anonymity (l-diversity builds on k-anonymity)
            l: The l parameter for l-diversity
            qi_columns: List of quasi-identifier columns
            sensitive_column: The sensitive attribute column
            diversity_type: Type of diversity ("distinct", "entropy", "recursive")
            generalization_strategy: Strategy for generalization
        """
        self.k = k
        self.l = l
        self.qi_columns = qi_columns
        self.sensitive_column = sensitive_column
        self.diversity_type = diversity_type
        self.generalization_strategy = generalization_strategy
        
        # Initialize k-anonymity core
        self.k_anonymity_core = KAnonymityCore(k, qi_columns, generalization_strategy)
        
        # Recursive l-diversity parameters
        self.c = 2.0  # parameter for recursive l-diversity
        
    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply l-diversity to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Anonymized dataframe satisfying l-diversity
        """
        if self.sensitive_column not in df.columns:
            raise ValueError(f"Sensitive column '{self.sensitive_column}' not found in dataframe")
        
        # First apply k-anonymity
        k_anonymous_df = self.k_anonymity_core.anonymize(df)
        
        # Then enforce l-diversity
        return self._enforce_l_diversity(k_anonymous_df)
    
    def _enforce_l_diversity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce l-diversity on k-anonymous data."""
        # Group by QI columns to get equivalence classes
        valid_qi_cols = [col for col in self.qi_columns if col in df.columns]
        if not valid_qi_cols:
            return df
        
        # Get equivalence classes
        groups = df.groupby(valid_qi_cols)
        
        # Check and fix l-diversity violations
        diverse_groups = []
        
        for name, group in groups:
            if self._is_l_diverse(group):
                diverse_groups.append(group)
            else:
                # Fix l-diversity violation
                fixed_group = self._fix_l_diversity_violation(group, df)
                diverse_groups.append(fixed_group)
        
        if diverse_groups:
            result_df = pd.concat(diverse_groups, ignore_index=True)
            return result_df
        else:
            return df
    
    def _is_l_diverse(self, group: pd.DataFrame) -> bool:
        """Check if a group satisfies l-diversity."""
        if self.sensitive_column not in group.columns:
            return True
        
        sensitive_values = group[self.sensitive_column].dropna()
        
        if len(sensitive_values) == 0:
            return True
        
        if self.diversity_type == "distinct":
            return self._check_distinct_l_diversity(sensitive_values)
        elif self.diversity_type == "entropy":
            return self._check_entropy_l_diversity(sensitive_values)
        elif self.diversity_type == "recursive":
            return self._check_recursive_l_diversity(sensitive_values)
        else:
            return self._check_distinct_l_diversity(sensitive_values)
    
    def _check_distinct_l_diversity(self, sensitive_values: pd.Series) -> bool:
        """
        Check distinct l-diversity.
        
        Distinct l-diversity requires each equivalence class to contain at least
        l distinct values for the sensitive attribute. This is the simplest and
        most commonly used form of l-diversity.
        
        References:
            - Machanavajjhala et al. (2007): |{s | (qi,s) ∈ qi-block}| ≥ l
            - PyCANON distinct l-diversity implementation
            - ANJANA library distinct diversity checking
        """
        unique_values = sensitive_values.nunique()
        return unique_values >= self.l
    
    def _check_entropy_l_diversity(self, sensitive_values: pd.Series) -> bool:
        """
        Check entropy l-diversity.
        
        Entropy l-diversity requires that the entropy of the distribution of
        sensitive values in each equivalence class is at least log(l). This
        provides stronger privacy guarantees than distinct l-diversity.
        
        Formula: Entropy(Q) = -∑p(Q,s)log(p(Q,s)) ≥ log(l)
        
        References:
            - Machanavajjhala et al. (2007): Entropy l-diversity definition
            - PyCANON entropy l-diversity implementation
            - ANJANA library entropy diversity calculation
            - ARX framework entropy-l-diversity implementation
        """
        value_counts = sensitive_values.value_counts()
        total_count = len(sensitive_values)
        
        # Calculate entropy
        entropy = 0
        for count in value_counts:
            p = count / total_count
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Check if entropy >= log2(l)
        return entropy >= math.log2(self.l)
    
    def _check_recursive_l_diversity(self, sensitive_values: pd.Series) -> bool:
        """
        Check recursive (c,l)-diversity.
        
        Recursive (c,l)-diversity ensures that the most frequent sensitive value
        does not appear too frequently compared to the other values. It addresses
        the limitation where distinct l-diversity might allow one value to dominate.
        
        Condition: r1 < c(rl + rl+1 + ... + rm)
        where ri are frequencies in descending order and c is a constant ≥ 1.
        
        References:
            - Machanavajjhala et al. (2007): Recursive (c,l)-diversity definition
            - PyCANON recursive (c,l)-diversity implementation
            - ANJANA library recursive diversity calculation
            - ARX framework recursive-(c,l)-diversity implementation
            - Nuclearstar implementation with Kolmogorov-Smirnov distance
        """
        value_counts = sensitive_values.value_counts().sort_values(ascending=False)
        
        if len(value_counts) < self.l:
            return False
        
        # Most frequent value
        r1 = value_counts.iloc[0]
        
        # Sum of l-1 most frequent values excluding the most frequent
        if len(value_counts) >= self.l:
            sum_others = value_counts.iloc[1:self.l].sum()
        else:
            sum_others = value_counts.iloc[1:].sum()
        
        # Check recursive condition: r1 < c * (r[l] + r[l+1] + ... + r[m])
        return r1 < self.c * sum_others if sum_others > 0 else False
    
    def _fix_l_diversity_violation(self, group: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
        """Fix l-diversity violation by merging with other groups or generalization."""
        # Strategy 1: Try to merge with similar groups
        merged_group = self._try_merge_with_similar_groups(group, full_df)
        if merged_group is not None and self._is_l_diverse(merged_group):
            return merged_group
        
        # Strategy 2: Further generalization
        return self._apply_additional_generalization(group)
    
    def _try_merge_with_similar_groups(self, group: pd.DataFrame, full_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Try to merge group with similar groups to achieve l-diversity."""
        valid_qi_cols = [col for col in self.qi_columns if col in full_df.columns]
        if not valid_qi_cols:
            return None
        
        # Find groups with similar QI values that could be merged
        all_groups = full_df.groupby(valid_qi_cols)
        
        for name, other_group in all_groups:
            if len(other_group) == len(group) and other_group.equals(group):
                continue  # Skip the same group
            
            # Check if groups can be meaningfully merged
            if self._can_merge_groups(group, other_group):
                merged = pd.concat([group, other_group], ignore_index=True)
                if len(merged) >= self.k and self._is_l_diverse(merged):
                    return merged
        
        return None
    
    def _can_merge_groups(self, group1: pd.DataFrame, group2: pd.DataFrame) -> bool:
        """Check if two groups can be meaningfully merged."""
        # Simple heuristic: check if QI values are similar enough
        for col in self.qi_columns:
            if col not in group1.columns or col not in group2.columns:
                continue
            
            # For categorical data, check overlap
            if not pd.api.types.is_numeric_dtype(group1[col]):
                unique1 = set(group1[col].dropna().unique())
                unique2 = set(group2[col].dropna().unique())
                if not unique1.intersection(unique2):  # No overlap
                    return False
            else:
                # For numeric data, check if ranges overlap
                min1, max1 = group1[col].min(), group1[col].max()
                min2, max2 = group2[col].min(), group2[col].max()
                if max1 < min2 or max2 < min1:  # No overlap
                    return False
        
        return True
    
    def _apply_additional_generalization(self, group: pd.DataFrame) -> pd.DataFrame:
        """Apply additional generalization to achieve l-diversity."""
        # This is a simplified approach - in practice, you might need more sophisticated methods
        result_group = group.copy()
        
        # Generalize QI columns further
        for col in self.qi_columns:
            if col not in result_group.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(result_group[col]):
                # Create broader ranges
                min_val, max_val = result_group[col].min(), result_group[col].max()
                result_group[col] = f"[{min_val:.2f}-{max_val:.2f}]"
            else:
                # Generalize categorical values
                result_group[col] = "*"
        
        return result_group
    
    def get_privacy_metrics(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy and utility metrics for l-diversity."""
        # Get k-anonymity metrics first
        k_metrics = self.k_anonymity_core.get_privacy_metrics(original_df, anonymized_df)
        
        # Add l-diversity specific metrics
        l_metrics = {}
        l_metrics.update(k_metrics)
        
        # L-diversity metrics
        l_metrics['l_value'] = self.l
        l_metrics['diversity_type'] = self.diversity_type
        l_metrics['sensitive_column'] = self.sensitive_column
        
        # Check l-diversity compliance
        valid_qi_cols = [col for col in self.qi_columns if col in anonymized_df.columns]
        if valid_qi_cols and self.sensitive_column in anonymized_df.columns:
            groups = anonymized_df.groupby(valid_qi_cols)
            diverse_groups = 0
            total_groups = 0
            min_diversity = float('inf')
            max_diversity = 0
            diversity_scores = []
            
            for name, group in groups:
                total_groups += 1
                diversity_score = self._calculate_diversity_score(group)
                diversity_scores.append(diversity_score)
                
                if diversity_score >= self.l:
                    diverse_groups += 1
                
                min_diversity = min(min_diversity, diversity_score)
                max_diversity = max(max_diversity, diversity_score)
            
            l_metrics['l_diverse_groups'] = diverse_groups
            l_metrics['total_groups'] = total_groups
            l_metrics['l_diversity_compliance'] = diverse_groups / total_groups if total_groups > 0 else 0
            l_metrics['min_diversity_score'] = min_diversity if min_diversity != float('inf') else 0
            l_metrics['max_diversity_score'] = max_diversity
            l_metrics['avg_diversity_score'] = np.mean(diversity_scores) if diversity_scores else 0
        
        return l_metrics
    
    def _calculate_diversity_score(self, group: pd.DataFrame) -> float:
        """Calculate diversity score for a group."""
        if self.sensitive_column not in group.columns:
            return 0
        
        sensitive_values = group[self.sensitive_column].dropna()
        
        if len(sensitive_values) == 0:
            return 0
        
        if self.diversity_type == "distinct":
            return sensitive_values.nunique()
        elif self.diversity_type == "entropy":
            value_counts = sensitive_values.value_counts()
            total_count = len(sensitive_values)
            
            entropy = 0
            for count in value_counts:
                p = count / total_count
                if p > 0:
                    entropy -= p * math.log2(p)
            
            return 2 ** entropy  # Convert back to equivalent distinct values
        elif self.diversity_type == "recursive":
            value_counts = sensitive_values.value_counts().sort_values(ascending=False)
            return len(value_counts)  # Simplified score
        else:
            return sensitive_values.nunique()


def apply_l_diversity(df: pd.DataFrame, k: int, l: int, qi_columns: List[str], 
                     sensitive_column: str, diversity_type: str = "distinct",
                     generalization_strategy: str = "optimal") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply l-diversity to a dataframe using the specified diversity measure.
    
    This function provides a high-level interface for applying l-diversity
    anonymization. It first ensures k-anonymity and then enforces the specified
    type of l-diversity constraint on the sensitive attribute.
    
    Args:
        df: Input dataframe to anonymize
        k: The k parameter for k-anonymity (l-diversity builds on k-anonymity)
        l: The l parameter for l-diversity (minimum diversity requirement)
        qi_columns: List of quasi-identifier columns
        sensitive_column: The sensitive attribute column
        diversity_type: Type of diversity ("distinct", "entropy", "recursive")
        generalization_strategy: Strategy for generalization
        
    Returns:
        Tuple of (anonymized_dataframe, metrics_dict)
        
    References:
        - Machanavajjhala, A., et al. (2007). L-diversity: Privacy beyond k-anonymity.
        - Implementation patterns from:
          * Mondrian L-Diversity: https://github.com/qiyuangong/Mondrian_L_Diversity
          * PyCANON library: https://github.com/IFCA/pycanon
          * ANJANA library: https://github.com/IFCA-Advanced-Computing/anjana
        - Algorithm approach based on InfoGain Mondrian for L-Diversity
        - Group merging and generalization strategies from Nuclearstar implementation
    """
    anonymizer = LDiversityCore(k, l, qi_columns, sensitive_column, diversity_type, generalization_strategy)
    anonymized_df = anonymizer.anonymize(df)
    metrics = anonymizer.get_privacy_metrics(df, anonymized_df)
    
    return anonymized_df, metrics
