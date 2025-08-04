# filepath: c:\Users\alise\OneDrive\Desktop\Bachelor Thesis\ml_models\src\anonymizers\mondrian_anonymizer.py
import pandas as pd
import numpy as np
import math
import os # For standalone testing

class Partition:
    """Represents an active or leaf partition of the dataset."""
    def __init__(self, idx: np.ndarray):
        self.idx = idx  # numpy array of actual DataFrame index values

def _choose_split_column(df: pd.DataFrame, part_idx: np.ndarray, qi_cols: list, numeric_qis: list) -> str | None:
    """Pick the attribute with the largest normalised range / diversity within the partition."""
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
    """Return two arrays of DataFrame index values after splitting on *col*."""
    if not len(part_idx): return None
    
    partition_df_view = df.loc[part_idx] # Work with the view of the current partition
    sub_col_data = partition_df_view[col]

    if sub_col_data.nunique() <= 1: # Cannot split if only one unique value or all same
        return None

    if col in numeric_qis:
        median = sub_col_data.median()
        left_mask = sub_col_data <= median
        right_mask = sub_col_data > median
        # If median split results in one empty partition (e.g. median is min or max)
        # and there are multiple unique values, try a more balanced split.
        if (not left_mask.any() or not right_mask.any()) and sub_col_data.nunique() > 1:
            sorted_unique_vals = sorted(sub_col_data.unique())
            # Split at the middle unique value if possible
            split_val = sorted_unique_vals[len(sorted_unique_vals) // 2 -1] if len(sorted_unique_vals) > 1 else sorted_unique_vals[0]
            left_mask = sub_col_data <= split_val
            right_mask = sub_col_data > split_val
            # Ensure both partitions are still non-empty after this fallback
            if not left_mask.any() or not right_mask.any(): return None

    else: # Categorical
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
    """Generalize QI values within the finalized partition in df_anonymized."""
    if not len(part_idx): return

    for col in qi_cols:
        if col not in df_anonymized.columns: continue
        
        subset_col_data = df_anonymized.loc[part_idx, col]
        if subset_col_data.empty: continue

        if col in numeric_qis:
            min_val, max_val = subset_col_data.min(), subset_col_data.max()
            # Apply to the original DataFrame using .loc
            df_anonymized.loc[part_idx, col] = f"[{min_val},{max_val}]" if min_val != max_val else str(min_val)
        else:
            unique_vals = sorted(list(subset_col_data.unique()))
            df_anonymized.loc[part_idx, col] = "{" + ",".join(map(str, unique_vals)) + "}"


def run_mondrian(df_input: pd.DataFrame, qi_cols: list, numeric_qis: list, k: int, sa_col: str | None = None) -> pd.DataFrame:
    """
    Applies Mondrian k-anonymity to the provided DataFrame.
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
    queue = [Partition(df_anonymized.index.values)] 
    final_partitions_indices = [] # Stores arrays of index values for final partitions

    while queue:
        node = queue.pop(0)
        current_part_idx = node.idx # These are actual DataFrame index values

        if not len(current_part_idx): continue

        # Stop condition: if partition is too small to split into two k-children,
        # or if it's already acceptably k-anonymous (k <= size < 2k)
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
            final_partitions_indices.append(current_part_idx)
            continue
        
        split_col = _choose_split_column(df_anonymized, current_part_idx, actual_qi_cols, actual_numeric_qis)

        if split_col is None: # No suitable column to split on (e.g., all values same within QIs)
            final_partitions_indices.append(current_part_idx)
            continue

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
            final_partitions_indices.append(current_part_idx) # So, keep the parent partition
        else:
            queue.append(Partition(left_child_idx))
            queue.append(Partition(right_child_idx))

    # Anonymize all finalized partitions
    for part_idx_to_anonymize in final_partitions_indices:
        if len(part_idx_to_anonymize) >= k: # Only generalize partitions that meet k
             _anonymise_partition(df_anonymized, part_idx_to_anonymize, actual_qi_cols, actual_numeric_qis)
        # else: (Handled by the warning above, or decide on suppression/other strategy)

    return df_anonymized

# --- Standalone testing block ---
if __name__ == "__main__":
    print(f"Running {__file__} as a standalone script for testing Mondrian logic...")
    
    # Create a sample DataFrame for testing
    data = {
        'age': [25, 30, 22, 45, 50, 35, 28, 60, 55, 40],
        'job': ['Dev', 'Dev', 'Analyst', 'Manager', 'Manager', 'Analyst', 'Dev', 'CEO', 'CEO', 'Manager'],
        'city': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
        'salary': [50, 60, 45, 80, 90, 70, 55, 120, 110, 75]
    }
    sample_df = pd.DataFrame(data)
    
    print("\n--- Original Sample DataFrame ---")
    print(sample_df)
    
    test_qi_cols = ['age', 'job', 'city']
    test_numeric_qis = ['age']
    test_k = 3
    
    print(f"\n--- Running Mondrian with k={test_k} ---")
    print(f"QIs: {test_qi_cols}, Numeric QIs: {test_numeric_qis}")
    
    anonymized_df = run_mondrian(sample_df.copy(), test_qi_cols, test_numeric_qis, test_k)
    
    print("\n--- Anonymized Sample DataFrame ---")
    print(anonymized_df)

    # Test with adult dataset if available (adjust path as needed)
    # Note: This part requires the adult_train_for_arx.csv to be accessible
    # For a truly standalone module, you might comment this out or make it more robust to file absence.
    try:
        # Path relative to this script's location in src/anonymizers/
        adult_csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "adult_train_for_arx.csv")
        if os.path.exists(adult_csv_path):
            print(f"\n--- Testing with Adult Dataset (path: {adult_csv_path}) ---")
            df_adult_raw = pd.read_csv(adult_csv_path, sep=';')
            adult_qi_cols = ["age", "workclass", "education", "maritalstatus", "occupation", "relationship", "race", "sex", "nativecountry"]
            adult_numeric_qis = ["age"]
            adult_k = 10
            
            print(f"Running Mondrian on Adult data with k={adult_k}...")
            adult_anonymized = run_mondrian(df_adult_raw.copy(), adult_qi_cols, adult_numeric_qis, adult_k)
            print("\n--- Anonymized Adult DataFrame (head) ---")
            print(adult_anonymized.head())
        else:
            print(f"\nAdult dataset not found at {adult_csv_path}, skipping Adult data test.")
    except Exception as e:
        print(f"Error during Adult dataset test: {e}")