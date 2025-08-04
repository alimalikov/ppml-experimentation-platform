"""
Simple working k-anonymity implementation for debugging purposes.
This will replace the problematic k_anonymity_core temporarily.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

def apply_k_anonymity(df: pd.DataFrame, k: int, qi_columns: List[str], 
                     generalization_strategy: str = "optimal") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simple k-anonymity implementation that actually works.
    """
    print(f"DEBUG: Starting k-anonymity with k={k}, columns={qi_columns}")
    
    if not qi_columns:
        return df.copy(), {"error": "No QI columns provided"}
    
    # Work with a copy
    result_df = df.copy()
    
    # Check initial equivalence classes
    if qi_columns:
        initial_groups = df.groupby(qi_columns).size()
        min_group_size = initial_groups.min()
        print(f"DEBUG: Initial min group size: {min_group_size}")
        
        if min_group_size >= k:
            print(f"DEBUG: Already {k}-anonymous!")
            return result_df, {
                "k_value": k,
                "min_group_size": int(min_group_size),
                "anonymized_equivalence_classes": len(initial_groups),
                "information_loss": 0.0
            }
    
    # Apply generalization to numeric columns
    for col in qi_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            print(f"DEBUG: Generalizing numeric column {col}")
            result_df[col] = generalize_numeric_column(df[col], k)
    
    # Check final equivalence classes
    final_groups = result_df.groupby(qi_columns).size()
    min_final_size = final_groups.min()
    
    print(f"DEBUG: Final min group size: {min_final_size}")
    
    # Calculate information loss
    original_unique = len(df.drop_duplicates(subset=qi_columns))
    final_unique = len(result_df.drop_duplicates(subset=qi_columns))
    info_loss = ((original_unique - final_unique) / original_unique) * 100 if original_unique > 0 else 0
    
    metrics = {
        "k_value": k,
        "min_group_size": int(min_final_size),
        "max_group_size": int(final_groups.max()),
        "avg_group_size": float(final_groups.mean()),
        "anonymized_equivalence_classes": len(final_groups),
        "information_loss": round(info_loss, 2),
        "suppression_ratio": 0.0
    }
    
    print(f"DEBUG: Metrics: {metrics}")
    return result_df, metrics

def generalize_numeric_column(series: pd.Series, k: int) -> pd.Series:
    """
    Generalize a numeric column to achieve better k-anonymity.
    """
    if series.empty:
        return series
    
    # Create ranges based on quantiles
    # More aggressive generalization for higher k values
    if k <= 3:
        # Use quartiles for low k
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    elif k <= 10:
        # Use fewer bins for medium k
        quantiles = [0, 0.33, 0.67, 1.0]
    else:
        # Use very few bins for high k
        quantiles = [0, 0.5, 1.0]
    
    # Calculate quantile values
    quantile_values = series.quantile(quantiles).unique()
    quantile_values = np.sort(quantile_values)
    
    # Create generalized ranges
    result = series.copy()
    
    for i in range(len(quantile_values) - 1):
        lower = quantile_values[i]
        upper = quantile_values[i + 1]
        
        # Create mask for values in this range
        if i == len(quantile_values) - 2:  # Last range, include upper bound
            mask = (series >= lower) & (series <= upper)
        else:
            mask = (series >= lower) & (series < upper)
        
        # Assign range string
        range_str = f"[{lower:.2f}-{upper:.2f}]"
        result[mask] = range_str
    
    return result
