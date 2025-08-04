"""
Core p-sensitive k-anonymity implementation.
This module provides p-sensitive k-anonymity algorithms for protecting against probabilistic attacks.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import warnings
from collections import Counter
import math
from .k_anonymity_core import KAnonymityCore

warnings.filterwarnings('ignore')

class PSensitiveCore:
    """
    Core p-sensitive k-anonymity implementation.
    """
    
    def __init__(self, k: int, p: float, qi_columns: List[str], sensitive_column: str,
                 sensitive_values: Optional[List[str]] = None, generalization_strategy: str = "optimal"):
        """
        Initialize p-sensitive k-anonymity core.
        
        Args:
            k: The k parameter for k-anonymity
            p: The p parameter for p-sensitivity (probability threshold)
            qi_columns: List of quasi-identifier columns
            sensitive_column: The sensitive attribute column
            sensitive_values: List of sensitive values to protect (if None, auto-detect)
            generalization_strategy: Strategy for generalization
        """
        self.k = k
        self.p = p
        self.qi_columns = qi_columns
        self.sensitive_column = sensitive_column
        self.sensitive_values = sensitive_values
        self.generalization_strategy = generalization_strategy
        
        # Initialize k-anonymity core
        self.k_anonymity_core = KAnonymityCore(k, qi_columns, generalization_strategy)
        
        # Auto-detected sensitive values
        self.auto_detected_sensitive_values = None
        
    def fit_sensitive_values(self, df: pd.DataFrame) -> None:
        """Auto-detect sensitive values if not provided."""
        if self.sensitive_values is None:
            if self.sensitive_column not in df.columns:
                raise ValueError(f"Sensitive column '{self.sensitive_column}' not found in dataframe")
            
            # Auto-detect sensitive values (minority classes or user-specified patterns)
            sensitive_data = df[self.sensitive_column].dropna()
            value_counts = sensitive_data.value_counts(normalize=True)
            
            # Consider values with frequency < 0.3 as potentially sensitive
            # or the least frequent value if all are common
            threshold = 0.3
            sensitive_candidates = value_counts[value_counts < threshold].index.tolist()
            
            if not sensitive_candidates:
                # If no rare values, consider the least frequent one
                sensitive_candidates = [value_counts.index[-1]]
            
            self.auto_detected_sensitive_values = sensitive_candidates
        else:
            self.auto_detected_sensitive_values = self.sensitive_values
    
    def get_sensitive_values(self) -> List[str]:
        """Get the list of sensitive values being protected."""
        return self.auto_detected_sensitive_values or self.sensitive_values or []
    
    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply p-sensitive k-anonymity to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Anonymized dataframe satisfying p-sensitive k-anonymity
        """
        if self.sensitive_column not in df.columns:
            raise ValueError(f"Sensitive column '{self.sensitive_column}' not found in dataframe")
        
        # Auto-detect sensitive values if needed
        self.fit_sensitive_values(df)
        
        # First apply k-anonymity
        k_anonymous_df = self.k_anonymity_core.anonymize(df)
        
        # Then enforce p-sensitivity
        return self._enforce_p_sensitivity(k_anonymous_df)
    
    def _enforce_p_sensitivity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce p-sensitivity on k-anonymous data."""
        # Group by QI columns to get equivalence classes
        valid_qi_cols = [col for col in self.qi_columns if col in df.columns]
        if not valid_qi_cols:
            return df
        
        # Get equivalence classes
        groups = df.groupby(valid_qi_cols)
        
        # Check and fix p-sensitivity violations
        sensitive_groups = []
        
        for name, group in groups:
            if self._is_p_sensitive(group):
                sensitive_groups.append(group)
            else:
                # Fix p-sensitivity violation
                fixed_group = self._fix_p_sensitivity_violation(group, df)
                sensitive_groups.append(fixed_group)
        
        if sensitive_groups:
            result_df = pd.concat(sensitive_groups, ignore_index=True)
            return result_df
        else:
            return df
    
    def _is_p_sensitive(self, group: pd.DataFrame) -> bool:
        """Check if a group satisfies p-sensitive k-anonymity."""
        if self.sensitive_column not in group.columns:
            return True
        
        sensitive_values_to_check = self.get_sensitive_values()
        if not sensitive_values_to_check:
            return True  # No sensitive values to protect
        
        sensitive_data = group[self.sensitive_column].dropna()
        
        if len(sensitive_data) == 0:
            return True
        
        # Check each sensitive value
        for sensitive_val in sensitive_values_to_check:
            sensitive_count = (sensitive_data == sensitive_val).sum()
            total_count = len(sensitive_data)
            
            if total_count > 0:
                probability = sensitive_count / total_count
                
                # The probability of inferring the sensitive value should be <= p
                if probability > self.p:
                    return False
        
        return True
    
    def _fix_p_sensitivity_violation(self, group: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
        """Fix p-sensitivity violation."""
        # Strategy 1: Try to merge with other groups
        merged_group = self._try_merge_for_p_sensitivity(group, full_df)
        if merged_group is not None and self._is_p_sensitive(merged_group):
            return merged_group
        
        # Strategy 2: Apply record modification or suppression
        return self._apply_p_sensitivity_modification(group)
    
    def _try_merge_for_p_sensitivity(self, group: pd.DataFrame, full_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Try to merge group with others to achieve p-sensitivity."""
        valid_qi_cols = [col for col in self.qi_columns if col in full_df.columns]
        if not valid_qi_cols:
            return None
        
        all_groups = full_df.groupby(valid_qi_cols)
        
        for name, other_group in all_groups:
            if len(other_group) == len(group) and other_group.equals(group):
                continue  # Skip the same group
            
            # Try merging and check if it satisfies p-sensitivity
            merged = pd.concat([group, other_group], ignore_index=True)
            if len(merged) >= self.k and self._is_p_sensitive(merged):
                return merged
        
        return None
    
    def _apply_p_sensitivity_modification(self, group: pd.DataFrame) -> pd.DataFrame:
        """Apply modifications to achieve p-sensitivity."""
        result_group = group.copy()
        sensitive_values_to_check = self.get_sensitive_values()
        
        if not sensitive_values_to_check:
            return result_group
        
        sensitive_data = result_group[self.sensitive_column].dropna()
        
        if len(sensitive_data) == 0:
            return result_group
        
        # For each violating sensitive value, reduce its frequency
        for sensitive_val in sensitive_values_to_check:
            sensitive_count = (sensitive_data == sensitive_val).sum()
            total_count = len(sensitive_data)
            
            if total_count > 0:
                current_probability = sensitive_count / total_count
                
                if current_probability > self.p:
                    # Calculate how many records need to be modified
                    target_count = int(self.p * total_count)
                    records_to_modify = sensitive_count - target_count
                    
                    # Apply modification strategies
                    result_group = self._modify_sensitive_records(
                        result_group, sensitive_val, records_to_modify
                    )
        
        return result_group
    
    def _modify_sensitive_records(self, group: pd.DataFrame, sensitive_val: str, 
                                count_to_modify: int) -> pd.DataFrame:
        """Modify records with sensitive values to achieve p-sensitivity."""
        if count_to_modify <= 0:
            return group
        
        result_group = group.copy()
        
        # Find records with the sensitive value
        sensitive_records = result_group[result_group[self.sensitive_column] == sensitive_val]
        
        if len(sensitive_records) <= count_to_modify:
            # Modify all records with this sensitive value
            records_to_modify = sensitive_records
        else:
            # Randomly select records to modify
            records_to_modify = sensitive_records.sample(n=count_to_modify)
        
        # Apply modification strategy
        modification_strategy = self._choose_modification_strategy(result_group, sensitive_val)
        
        for idx in records_to_modify.index:
            if modification_strategy == "suppress":
                # Remove the record (if it doesn't violate k-anonymity)
                if len(result_group) - 1 >= self.k:
                    result_group = result_group.drop(idx)
            elif modification_strategy == "generalize":
                # Generalize the sensitive value
                result_group.loc[idx, self.sensitive_column] = "*"
            elif modification_strategy == "substitute":
                # Substitute with a non-sensitive value
                non_sensitive_values = self._get_non_sensitive_values(result_group)
                if non_sensitive_values:
                    substitute_value = np.random.choice(non_sensitive_values)
                    result_group.loc[idx, self.sensitive_column] = substitute_value
        
        return result_group
    
    def _choose_modification_strategy(self, group: pd.DataFrame, sensitive_val: str) -> str:
        """Choose the best modification strategy based on the data and context."""
        # Simple heuristic: prefer substitution, then generalization, then suppression
        non_sensitive_values = self._get_non_sensitive_values(group)
        
        if non_sensitive_values:
            return "substitute"
        elif len(group) > self.k:
            return "suppress"
        else:
            return "generalize"
    
    def _get_non_sensitive_values(self, group: pd.DataFrame) -> List[str]:
        """Get list of non-sensitive values from the group."""
        if self.sensitive_column not in group.columns:
            return []
        
        all_values = group[self.sensitive_column].dropna().unique().tolist()
        sensitive_values_to_check = self.get_sensitive_values()
        
        non_sensitive = [val for val in all_values if val not in sensitive_values_to_check]
        return non_sensitive
    
    def get_privacy_metrics(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy and utility metrics for p-sensitive k-anonymity."""
        # Get k-anonymity metrics first
        k_metrics = self.k_anonymity_core.get_privacy_metrics(original_df, anonymized_df)
        
        # Add p-sensitive specific metrics
        p_metrics = {}
        p_metrics.update(k_metrics)
        
        # P-sensitive metrics
        p_metrics['p_value'] = self.p
        p_metrics['sensitive_column'] = self.sensitive_column
        p_metrics['protected_sensitive_values'] = self.get_sensitive_values()
        
        # Check p-sensitivity compliance
        valid_qi_cols = [col for col in self.qi_columns if col in anonymized_df.columns]
        if valid_qi_cols and self.sensitive_column in anonymized_df.columns:
            groups = anonymized_df.groupby(valid_qi_cols)
            p_sensitive_groups = 0
            total_groups = 0
            max_probability = 0
            min_probability = 1.0
            probability_scores = []
            
            for name, group in groups:
                total_groups += 1
                max_prob_in_group = self._calculate_max_sensitive_probability(group)
                probability_scores.append(max_prob_in_group)
                
                if max_prob_in_group <= self.p:
                    p_sensitive_groups += 1
                
                max_probability = max(max_probability, max_prob_in_group)
                min_probability = min(min_probability, max_prob_in_group)
            
            p_metrics['p_sensitive_groups'] = p_sensitive_groups
            p_metrics['total_groups'] = total_groups
            p_metrics['p_sensitivity_compliance'] = p_sensitive_groups / total_groups if total_groups > 0 else 0
            p_metrics['max_sensitive_probability'] = max_probability
            p_metrics['min_sensitive_probability'] = min_probability
            p_metrics['avg_sensitive_probability'] = np.mean(probability_scores) if probability_scores else 0
        
        return p_metrics
    
    def _calculate_max_sensitive_probability(self, group: pd.DataFrame) -> float:
        """Calculate the maximum probability of inferring sensitive values in a group."""
        if self.sensitive_column not in group.columns:
            return 0
        
        sensitive_values_to_check = self.get_sensitive_values()
        if not sensitive_values_to_check:
            return 0
        
        sensitive_data = group[self.sensitive_column].dropna()
        
        if len(sensitive_data) == 0:
            return 0
        
        max_prob = 0
        for sensitive_val in sensitive_values_to_check:
            sensitive_count = (sensitive_data == sensitive_val).sum()
            probability = sensitive_count / len(sensitive_data)
            max_prob = max(max_prob, probability)
        
        return max_prob


def apply_p_sensitive_k_anonymity(df: pd.DataFrame, k: int, p: float, qi_columns: List[str], 
                                 sensitive_column: str, sensitive_values: Optional[List[str]] = None,
                                 generalization_strategy: str = "optimal") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply p-sensitive k-anonymity to a dataframe.
    
    Args:
        df: Input dataframe
        k: The k parameter for k-anonymity
        p: The p parameter for p-sensitivity
        qi_columns: List of quasi-identifier columns
        sensitive_column: The sensitive attribute column
        sensitive_values: List of sensitive values to protect (if None, auto-detect)
        generalization_strategy: Strategy for generalization
        
    Returns:
        Tuple of (anonymized_dataframe, metrics_dict)
    """
    anonymizer = PSensitiveCore(k, p, qi_columns, sensitive_column, sensitive_values, generalization_strategy)
    anonymized_df = anonymizer.anonymize(df)
    metrics = anonymizer.get_privacy_metrics(df, anonymized_df)
    
    return anonymized_df, metrics
