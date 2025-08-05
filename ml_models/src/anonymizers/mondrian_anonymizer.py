# Mondrian K-Anonymity Algorithm Implementation
# Based on: LeFevre, K., DeWitt, D.J., Ramakrishnan, R. (2006). 
# Mondrian multidimensional k-anonymity. ICDE 2006.
#
# Key references:
# - https://github.com/qiyuangong/Mondrian (most cited Python implementation)
# - https://github.com/Andrew0133/Mondrian-k-anonimity (clean implementation)
# - https://github.com/Nuclearstar/K-Anonymity (comprehensive with notebooks)
# - https://github.com/KatharinaMoel/Mondrian_py3 (Python 3 port)
#
# Algorithm: Top-down greedy partitioning using kd-tree approach
# - Recursively splits data into binary partitions
# - Uses median split for numeric, balanced split for categorical
# - Ensures each partition has >= k records for k-anonymity

import pandas as pd
import numpy as np
import math
import os # For standalone testing

class Partition:
    """Represents an active or leaf partition of the dataset."""
    def __init__(self, idx: np.ndarray):
        self.idx = idx  # numpy array of actual DataFrame index values

def _choose_split_column(df: pd.DataFrame, part_idx: np.ndarray, qi_cols: list, numeric_qis: list) -> str | None:
    """Pick the attribute with the largest normalised range / diversity within the partition.
    
    # Standard Mondrian heuristic: choose dimension with widest normalized range
    # Ref: qiyuangong/Mondrian, Andrew0133/Mondrian-k-anonimity
    # For numeric: range = max - min, for categorical: range = unique count
    """
    best_col, best_range = None, -math.inf
    if not len(part_idx) or df.loc[part_idx].empty:
        return None
    
    subset = df.loc[part_idx, qi_cols]
    if subset.empty:
        return None

    for col in qi_cols:
        if subset[col].empty: continue
        if col in numeric_qis:
            # Ensure there are values to compute range; handle single unique value case
            rng = subset[col].max() - subset[col].min() if subset[col].nunique() > 1 else 0
        else:
            rng = subset[col].nunique()
        
        if rng > best_range:
            best_col, best_range = col, rng
    
    return best_col if best_range > 0 else None # Only return if there's some diversity

def _split_partition(df: pd.DataFrame, part_idx: np.ndarray, col: str, numeric_qis: list) -> tuple[np.ndarray, np.ndarray] | None:
    """Return two arrays of DataFrame index values after splitting on *col*.
    
    # Mondrian splitting strategy:
    # - Numeric: median split (strict mode: no intersection)
    # - Categorical: balanced split by frequency
    # Ref: LeFevre et al. (2006), qiyuangong/Mondrian implementation
    """
    if not len(part_idx): return None
    
    partition_df_view = df.loc[part_idx] # Work with the view of the current partition
    sub_col_data = partition_df_view[col]

    if sub_col_data.nunique() <= 1: # Cannot split if only one unique value or all same
        return None

    if col in numeric_qis:
        # Median split for numeric attributes (standard Mondrian approach)
        # Ref: Andrew0133/Mondrian-k-anonimity median-partition strategy
        median = sub_col_data.median()
        left_mask = sub_col_data <= median
        right_mask = sub_col_data > median
        # If median split results in one empty partition (e.g. median is min or max)
        # and there are multiple unique values, try a more balanced split.
        if (not left_mask.any() or not right_mask.any()) and sub_col_data.nunique() > 1:
            # Fallback: split at middle unique value for better balance
            # Handles edge case where median == min or max value
            sorted_unique_vals = sorted(sub_col_data.unique())
            # Split at the middle unique value if possible
            split_val = sorted_unique_vals[len(sorted_unique_vals) // 2 -1] if len(sorted_unique_vals) > 1 else sorted_unique_vals[0]
            left_mask = sub_col_data <= split_val
            right_mask = sub_col_data > split_val
            # Ensure both partitions are still non-empty after this fallback
            if not left_mask.any() or not right_mask.any(): return None

    else: # Categorical
        # Balanced categorical split by frequency
        # Ref: Mondrian implementations for categorical attribute handling
        counts = sub_col_data.value_counts()
        # A simple split: take the most frequent category for one side, rest to other
        # More sophisticated balancing could be used if needed.
        target_half_count = len(sub_col_data) / 2
        current_sum = 0
        left_categories = []
        sorted_categories = counts.index.tolist()

        for cat_idx, cat_val in enumerate(sorted_categories):
            left_categories.append(cat_val)
            current_sum += counts[cat_val]
            # Ensure we don't put all categories in the left split if there's more than one
            if current_sum >= target_half_count and (cat_idx + 1) < len(sorted_categories):
                break
        
        left_mask = sub_col_data.isin(left_categories)
        right_mask = ~left_mask

    if not left_mask.any() or not right_mask.any(): # Check if any split actually happened
        return None

    # Get original DataFrame index values for the new partitions
    left_indices = partition_df_view.index[left_mask].values
    right_indices = partition_df_view.index[right_mask].values
    
    return left_indices, right_indices

def _anonymise_partition(df_anonymized: pd.DataFrame, part_idx: np.ndarray, qi_cols: list, numeric_qis: list):
    """Generalize QI values within the finalized partition in df_anonymized.
    
    # Standard Mondrian generalization:
    # - Numeric: create ranges [min,max]
    # - Categorical: create sets {val1,val2,...}
    # Ref: LeFevre et al. (2006), qiyuangong/Mondrian generalization strategy
    """
    if not len(part_idx): return

    for col in qi_cols:
        if col not in df_anonymized.columns: continue
        
        subset_col_data = df_anonymized.loc[part_idx, col]
        if subset_col_data.empty: continue

        if col in numeric_qis:
            # Create range generalization for numeric attributes
            min_val, max_val = subset_col_data.min(), subset_col_data.max()
            # Apply to the original DataFrame using .loc
            df_anonymized.loc[part_idx, col] = f"[{min_val},{max_val}]" if min_val != max_val else str(min_val)
        else:
            # Create set generalization for categorical attributes
            unique_vals = sorted(list(subset_col_data.unique()))
            df_anonymized.loc[part_idx, col] = "{" + ",".join(map(str, unique_vals)) + "}"


def run_mondrian(df_input: pd.DataFrame, qi_cols: list, numeric_qis: list, k: int, sa_col: str | None = None) -> pd.DataFrame:
    """
    Applies Mondrian k-anonymity to the provided DataFrame.
    
    # Top-down greedy Mondrian algorithm implementation
    # Ref: LeFevre, K., DeWitt, D.J., Ramakrishnan, R. (2006). Mondrian multidimensional k-anonymity
    # Implementation based on: https://github.com/qiyuangong/Mondrian (most cited)
    # 
    # Algorithm steps:
    # 1. Start with entire dataset as single partition
    # 2. Choose dimension with largest normalized range
    # 3. Split partition using median (numeric) or balanced (categorical)
    # 4. Ensure both children have >= k records
    # 5. Generalize final partitions
    
    Args:
        df_input (pd.DataFrame): The input DataFrame.
        qi_cols (list): List of quasi-identifier column names.
        numeric_qis (list): List of QI columns that should be treated as numeric.
        k (int): The k-anonymity parameter.
        sa_col (str | None, optional): Name of the sensitive attribute column (not directly used by Mondrian).
    Returns:
        pd.DataFrame: The k-anonymized DataFrame.
    """
    if df_input.empty:
        print("Error: Input DataFrame is empty for run_mondrian.")
        return pd.DataFrame()
    if not qi_cols:
        print("Error: No Quasi-identifier columns provided for run_mondrian.")
        return df_input.copy() # Return a copy if no QIs

    df_anonymized = df_input.copy()

    # Filter qi_cols and numeric_qis to only include columns present in the DataFrame
    actual_qi_cols = [col for col in qi_cols if col in df_anonymized.columns]
    if not actual_qi_cols:
        print("Error: None of the specified QI columns exist in the DataFrame.")
        return df_anonymized
    
    actual_numeric_qis = [col for col in numeric_qis if col in actual_qi_cols]

    # Initial partition contains all DataFrame index values
    # Using kd-tree approach: start with root containing entire dataset
    queue = [Partition(df_anonymized.index.values)] 
    final_partitions_indices = [] # Stores arrays of index values for final partitions

    while queue:
        node = queue.pop(0)
        current_part_idx = node.idx # These are actual DataFrame index values

        if not len(current_part_idx): continue

        # Stop condition: if partition is too small to split into two k-children,
        # or if it's already acceptably k-anonymous (k <= size < 2k)
        # Ref: Standard Mondrian stopping criteria from qiyuangong/Mondrian
        if len(current_part_idx) < k: # This partition is already smaller than k, problematic
            # For a robust implementation, such partitions might need special handling
            # (e.g., suppression or merging if possible, though merging is complex here)
            # For now, we'll add it to final partitions, but it won't satisfy k-anonymity.
            # Or, simply don't add it if we strictly want only >= k partitions.
            # Let's assume we want to process it for generalization as is, or log it.
            print(f"Warning: Partition of size {len(current_part_idx)} is smaller than k={k}.")
            final_partitions_indices.append(current_part_idx)
            continue

        if len(current_part_idx) < 2 * k : # Cannot split into two children of at least size k
            # Standard Mondrian: stop splitting when partition < 2k
            final_partitions_indices.append(current_part_idx)
            continue
        
        # Choose split dimension using standard Mondrian heuristic
        split_col = _choose_split_column(df_anonymized, current_part_idx, actual_qi_cols, actual_numeric_qis)

        if split_col is None: # No suitable column to split on (e.g., all values same within QIs)
            # "No allowable cut" case - all QI values identical in partition
            final_partitions_indices.append(current_part_idx)
            continue

        # Attempt binary split using median (numeric) or balanced (categorical)
        split_result = _split_partition(df_anonymized, current_part_idx, split_col, actual_numeric_qis)
        
        if split_result is None: # Split failed (e.g., couldn't create two non-empty children)
            final_partitions_indices.append(current_part_idx)
            continue
        
        left_child_idx, right_child_idx = split_result

        # Ensure children resulting from split are valid before checking k
        if not len(left_child_idx) or not len(right_child_idx):
            final_partitions_indices.append(current_part_idx) # Split was not effective
            continue

        if len(left_child_idx) < k or len(right_child_idx) < k: # Proposed split violates k-anonymity
            # K-constraint violation: keep parent partition instead of invalid children
            final_partitions_indices.append(current_part_idx) # So, keep the parent partition
        else:
            # Valid split: add children to queue for further processing
            queue.append(Partition(left_child_idx))
            queue.append(Partition(right_child_idx))

    # Anonymize all finalized partitions using standard generalization
    # Numeric: ranges [min,max], Categorical: sets {val1,val2,...}
    for part_idx_to_anonymize in final_partitions_indices:
        if len(part_idx_to_anonymize) >= k: # Only generalize partitions that meet k
             _anonymise_partition(df_anonymized, part_idx_to_anonymize, actual_qi_cols, actual_numeric_qis)
        # else: (Handled by the warning above, or decide on suppression/other strategy)

    return df_anonymized
